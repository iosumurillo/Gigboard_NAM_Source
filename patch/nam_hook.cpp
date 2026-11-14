// libnam_hook.so -- Neural Amp Modeler (NAM, MIT-licensed,
// https://github.com/sdatkinson/NeuralAmpModelerCore) inference, injected into
// HeadRush Evil.
//
// Three hook entry points, for three different designs (oldest to current):
//
//   nam_process(this)  -- SUPERSEDED reference implementation. Replaces
//     IRLoaderProcessThread::process() (vtable slot 3). Rejected as the
//     shipped design because IRLoaderProcessThread's vtable is shared by
//     every IR Loader slot -- patching it kills real .wav IR loading
//     everywhere, not just in one pedal. Kept only as a proven-working
//     reference/fallback. ABI: this+0x68=float* input, this+0x6c=float*
//     output, this+0x70=int32_t num_samples (mono, this-only, no extra args).
//
//   nam_process_gonk(...) -- CURRENT hijack design, name kept from its
//     original (superseded) Gonkulator target for minimal churn. Replaces a
//     REAL engine object's process() vtable slot in place, on the shared
//     class -- so every instance of that pedal type loses its real function
//     board-wide. Tried against Gonkulator (dead code, no UI page --
//     nam_process_gonk was simply never invoked), then Volume (works
//     mechanically, but rejected: user needs Volume's real function, tied
//     to the expression pedal). Now targets ANXIETY OD (v1) -- see
//     patch_gonkulator.py for the full derivation -- the user's own choice
//     of a pedal they're fine sacrificing board-wide. In the optional
//     "up to 4 instances" build, the SAME hook function is also wired to
//     Anxiety OD V2's separate engine vtable (own hijack, own hook_slot,
//     own env var -- see nam_preload.cpp) -- it dispatches per-instance by
//     engine-object pointer (state_for()) regardless of which pedal class
//     called in, so no V2-specific code exists here at all. ABI:
//       void process(EngineObj* this, uint32_t param2, float** input,
//                     uint32_t numChannels, float** output, int32_t numFrames,
//                     uint32_t* flags, void* ctx)
//     input/output are arrays of per-channel float* (numChannels is 1 or 2).
//
//   nam_process_naml(...) -- additive design, currently UNREACHABLE (no UI
//     path constructs its new pedal type/menu entry -- see README.md).
//     Same 8-arg ABI as nam_process_gonk (the new engine object is a
//     byte-for-byte clone of Gonkulator's own layout, see BREAKTHROUGH
//     #6/patch_namloader.py), but this is a genuinely NEW, independent
//     object graph -- whatever real class it's cloned from is never written
//     to. Also handles Input/Output trim (see nam_set_input_trim/
//     nam_set_output_trim below), which nam_process_gonk's hijack design
//     can't offer (no spare vtable slots on an existing, shared class).
//
// All three are loaded via dlopen() by nam_preload.cpp's constructor, which
// writes the corresponding function pointer into each trampoline's hook_slot
// (see trampoline.S/trampoline_gonk.S/trampoline_naml.S/trampoline_trim.S).
//
// Model loading happens once, off the audio thread, on first process() call.
// Until it completes, audio passes through dry (never silence, never garbage).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/resource.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

namespace
{

// Raw pthread_create/pthread_detach instead of std::thread everywhere in
// this file. This library is compiled with -static-libstdc++/-static-libgcc
// (needed: the real device's libstdc++.so.6 is a GCC ~10 build, GLIBCXX
// 3.4.28 max, too old for symbols this cross-toolchain's libstdc++
// generates -- confirmed, dynamic linking fails outright with "GLIBCXX_
// 3.4.32 not found"), but Evil itself and libnam_preload.so both use the
// SYSTEM libstdc++.so.6 dynamically. That split -- one self-contained
// static copy of libstdc++ internals in this .so, a separate dynamic copy
// shared by the rest of the process -- segfaults (confirmed via a minimal
// QEMU reproduction) the moment std::thread's constructor runs from inside
// this library: its internal thread bookkeeping isn't the same instance the
// rest of the process's pthread/exception machinery expects. Plain
// pthread_create is a C API with no such internal C++ static state to
// duplicate, so it sidesteps the conflict entirely.
template <typename F>
void spawn_detached(F&& fn)
{
  auto* ctx = new std::decay_t<F>(std::forward<F>(fn));
  auto* trampoline = +[](void* arg) -> void* {
    auto* f = static_cast<std::decay_t<F>*>(arg);
    (*f)();
    delete f;
    return nullptr;
  };
  pthread_t tid;
  if (pthread_create(&tid, nullptr, trampoline, ctx) != 0)
  {
    delete ctx;
    return;
  }
  pthread_detach(tid);
}

// TODO: confirm Evil's actual engine sample rate (48000 assumed; check
// /Engine/... property tree or IRLoaderController's Reset() call site).
constexpr double kSampleRate = 48000.0;

// -ffast-math's crtfastmath.o sets the VFP/NEON flush-to-zero bit, but only
// on whichever thread happens to run this .so's static initializers --
// typically whatever thread calls dlopen() on us, NOT Evil's own
// already-running real-time audio thread, and NOT any thread this library
// spawns itself via pthread_create (FPSCR/FPCR is per-thread state, never
// inherited -- confirmed as the exact mechanism behind
// https://github.com/mixxxdj/mixxx/issues/16126, an almost identical
// real-hardware symptom: "full-volume digital noise" once a missing FZ bit
// let a denormal stall a real-time audio callback past its deadline).
// Without this, a WaveNet's internal hidden state naturally decays toward
// (but never quite reaches) zero during the ~100ms duck-and-switch
// fade-to-silence and during the silent prewarm blocks below -- exactly
// where denormals are most likely -- and denormal float arithmetic on ARM
// takes a slow microcoded path, a real source of the residual switch-time
// noise this is meant to close (on top of the fade/prewarm already in
// place). Cheap enough (a couple of instructions) to just call
// unconditionally at the top of every function that runs DSP::process(),
// on every thread that might do so -- there's no "set once, applies
// everywhere" version of this bit.
inline void flush_denormals_to_zero()
{
#if defined(__arm__)
  uint32_t fpscr;
  asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
  fpscr |= (1u << 24); // FZ
  asm volatile("vmsr fpscr, %0" : : "r"(fpscr));
#elif defined(__aarch64__)
  uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= (1ull << 24); // FZ
  asm volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
  // Host/x86 test builds: deliberately a no-op -- SSE's own FTZ/DAZ
  // (MXCSR) is a different register this isn't attempting to manage, and
  // denormal stalls there don't reproduce the real-hardware symptom this
  // exists for.
}

// Every *.nam model this library ever constructs/prewarms/benchmarks
// (preload_models_in_background's validity check, and
// switch_model_in_background's full construct-calibrate-prewarm) happens on
// a thread WE spawned, competing for the same single ARM core as Evil's own
// real-time audio thread -- this device has no SMP to keep them apart.
// Reported on real hardware as several distinct glitches landing right at
// model construction time, both on the very first Anxiety OD engage (the
// first-ever model load) and on every later switch (which redoes this same
// construct-prewarm work for whichever model is newly selected, cache or no
// cache -- see g_calibration_cache's own comment on what IS cached).
// Lowering this thread's niceness costs nothing (this work is not latency-
// sensitive -- nothing is listening for its result except the fade logic,
// which is happy to wait) and gives the kernel's scheduler every reason to
// always favor Evil's audio thread under contention, on Linux this is
// per-thread state (setpriority(PRIO_PROCESS, 0, ...) with who=0 resolves to
// the CALLING thread's own kernel task, not the whole process -- unlike
// POSIX's nominal process-wide semantics), so this can never lower priority
// for Evil's own threads, only threads this library spawns itself.
inline void lower_background_thread_priority()
{
  struct sched_param sp;
  std::memset(&sp, 0, sizeof(sp));
  sched_setscheduler(0, SCHED_IDLE, &sp);
  setpriority(PRIO_PROCESS, 0, 19); // 19 = lowest (nicest) niceness
  nice(19);
}

// Folder scanned for *.nam files. Originally piggybacked "Impulse
// Responses" (matching how real IRs work), but Evil's own IR-folder sync
// logic purges anything it doesn't recognize as a real IR (i.e. non-.wav)
// whenever the USB drive gets reconnected -- confirmed on real hardware,
// .nam files placed there vanish the next time the USB transfer view
// reopens. Own dedicated sibling folder instead, untouched by that sync.
// Overridable via NAM_MODEL_DIR for host-side / emulated testing only.
std::filesystem::path model_dir()
{
  if (const char* env = std::getenv("NAM_MODEL_DIR"))
    return std::filesystem::path(env);
  return "/media/az01-internal/Evil/usb_mnt/NAM";
}

// Legacy single-fixed-file path, used only by the superseded nam_process()
// (IRLoader reference hook). Overridable via NAM_MODEL_PATH.
const char* model_path()
{
  if (const char* env = std::getenv("NAM_MODEL_PATH"))
    return env;
  return "/media/az01-internal/Evil/usb_mnt/NAM/model.nam";
}

// Knob raw-value range assumed for normalizing into a model index. Current
// hijack target is AnxietyOD -- Drive/Tone/Level are all plain 0-100%
// knobs (unlike Gonkulator's hijacked "Rate" knob, which was 200-2000 Hz as
// a ring-mod carrier frequency -- kept as a cautionary note: getting this
// range wrong silently clamps every real knob position to the same index,
// via std::clamp(t, 0.0f, 1.0f) in knob_value_to_index()). Override via
// NAM_KNOB_MIN/NAM_KNOB_MAX if this still isn't exactly right.
//
// CONFIRMED via live-hardware wide-memory diffing: the real range is
// [0.0, 1.0] (normalized), not [0, 100] -- an earlier assumption made when
// the (wrong) +0x2ac offset was still believed to be the model-select
// field. All three real knobs default to 0.5 and were observed ranging
// 0.0-1.0 when swept to their extremes.
float knob_min()
{
  if (const char* env = std::getenv("NAM_KNOB_MIN"))
    return std::strtof(env, nullptr);
  return 0.0f;
}

float knob_max()
{
  if (const char* env = std::getenv("NAM_KNOB_MAX"))
    return std::strtof(env, nullptr);
  return 1.0f;
}

// Moved here (from the Input/Output trim section further down) so
// nam_process_gonk can use these directly -- see its own comment on why
// Tone/Level are now read straight from the engine object instead of via a
// separate injected setter.
float trim_max_db()
{
  if (const char* env = std::getenv("NAM_TRIM_MAX_DB"))
    return std::strtof(env, nullptr);
  return 12.0f;
}

// raw01 is the knob's own [0,1] range directly (NOT pre-converted to
// bipolar) -- 0.0=true silence, 0.5=unity/0dB, 1.0=+trim_max_db() boost.
// Originally symmetric bipolar [-1,+1] mapped 0% to -trim_max_db() (still
// clearly audible, e.g. -12dB), not silence -- confirmed on real hardware
// this doesn't match the expected "0% = mute" behavior of a trim/volume
// control. Below center is a plain linear amplitude fade to true zero
// (matches how a physical volume knob feels down near its minimum); at or
// above center it's the original dB-boost curve.
float trim_gain(float raw01)
{
  if (raw01 <= 0.0f)
    return 0.0f;
  if (raw01 < 0.5f)
    return raw01 / 0.5f;
  const float bipolar_above_center = (raw01 - 0.5f) * 2.0f; // 0..1
  const float db = bipolar_above_center * trim_max_db();
  return std::pow(10.0f, db / 20.0f);
}

// Every *.nam file under model_dir() is read and JSON-parsed exactly ONCE,
// the first time Anxiety OD is actually engaged, and kept in memory for the
// rest of the boot -- model switching after that never touches the
// filesystem at all, it just re-constructs a nam::DSP from the already-
// in-memory JSON.
//
// Deliberately triggered LAZILY, not eagerly at Evil startup. Tried eager
// TWICE now, both reverted on real hardware -- see nam_preload.cpp's own
// comment (right where the eager call would go) for both failure modes:
// an unconditional-every-boot filesystem scan made the USB-transfer view
// hang worse, and separately, just spawning this thread that early (from
// the LD_PRELOAD constructor, before Evil's own main() runs) broke boot
// entirely. Keeping this conditional on real pedal use (as it always was
// pre-both-attempts) avoids both failure modes -- the "never re-scan on
// every switch" benefit (see g_preload_started below) doesn't need eager
// triggering to work, only lazy-but-cached.
struct CachedModel
{
  std::string display_name; // filename only, for sorting
  nlohmann::json config;
};

// Calibration result cache, indexed the same way as g_cached_models (one
// entry per cached model, sized once preload finishes). Kept separate from
// CachedModel itself since std::atomic isn't movable and g_cached_models is
// move-assigned as a whole when preload finishes. ratio<0.0 = "not yet
// calibrated for this model". Real hardware showed glitches/occasional
// full-device reboots when repeatedly switching models -- every switch was
// redoing the full quality-tier benchmark from scratch (which itself
// reconstructs the internal WaveNet sub-model 2x per channel, once per tier
// tested), real, avoidable CPU/memory churn on every single switch. Caching
// per-model means only the FIRST time a given model is selected pays that
// cost; switching back to an already-seen model is instant. A plain mutex
// is fine here (calibration is rare -- once per distinct model ever
// selected -- never on the real-time audio path).
//
// `epoch` records which g_topology_epoch generation a cached ratio was
// computed against. A ratio computed when only 2 instances shared the
// board's CPU budget is too generous once a 3rd/4th instance joins and the
// per-instance share shrinks -- see g_topology_epoch's own comment. Tagging
// each cache entry with the epoch it was computed under (instead of
// clearing the whole cache on every topology change, which would need a
// mutex lock from the audio thread) means the existing cache-miss path
// below transparently recomputes the FIRST time -- for any reason, a live
// switch or a topology-triggered forced reload -- any model is touched
// after the epoch it was cached under goes stale.
struct CalibrationEntry
{
  double ratio = -1.0;
  int epoch = -1;
};
std::vector<CalibrationEntry> g_calibration_cache;
std::mutex g_calibration_cache_mutex;

// Bumped every time state_for() claims a brand-new instance slot (i.e. a
// pedal instance calls in for the very first time this boot) -- see
// count_claimed_instances()/calibrate_slimmable_quality() for why the
// shared per-block CPU budget shrinks every time another instance joins.
// Lock-free (plain atomic increment, no mutex) so bumping it from the audio
// thread (state_for() runs there) never risks blocking on a background
// thread holding g_calibration_cache_mutex.
std::atomic<int> g_topology_epoch{0};

std::vector<CachedModel> g_cached_models;
std::atomic<bool> g_models_ready{false};
std::atomic<bool> g_preload_started{false};

void preload_models_in_background()
{
  bool expected = false;
  if (!g_preload_started.compare_exchange_strong(expected, true))
    return; // already started (or done) -- only ever run once per boot

  spawn_detached([]() {
    flush_denormals_to_zero(); // brand-new thread -- see its own comment
    lower_background_thread_priority(); // ditto -- see its own comment
    std::vector<CachedModel> found;
    std::error_code ec;
    const std::vector<std::filesystem::path> search_dirs = {
      model_dir(),
      "/media/az01-internal/Evil/usb_mnt/NAM",
      "/media/az01-internal/Evil/usb_mnt/nam",
      "/media/az01-internal/Evil/NAM",
      "/media/az01-internal/Evil/nam",
      "/media/az01-internal/NAM",
      "/NAM"
    };

    for (const auto& sdir : search_dirs)
    {
      if (!std::filesystem::exists(sdir, ec))
        continue;
      for (const auto& entry : std::filesystem::recursive_directory_iterator(
               sdir, std::filesystem::directory_options::skip_permission_denied, ec))
      {
        if (ec)
          break;
        if (!entry.is_regular_file(ec))
          continue;
        // Skip macOS AppleDouble sidecar files ("._realname.nam") -- Finder
        // creates these alongside real files when copying to a FAT/exFAT USB
        // drive (metadata/resource fork it can't store natively). They carry
        // the same .nam extension but binary, non-JSON content that throws
        // when parsed. They also sort first alphabetically ('.' < any
        // letter), so left unfiltered one would silently become index 0.
        const std::string filename = entry.path().filename().string();
        if (filename.rfind("._", 0) == 0)
          continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext != ".nam")
          continue;
        bool already = false;
        for (const auto& existing : found)
        {
          if (existing.display_name == filename)
          {
            already = true;
            break;
          }
        }
        if (already)
          continue;
        try
        {
          std::ifstream in(entry.path());
          nlohmann::json j;
          in >> j;
          // Fully test-construct the model here, not just parse its JSON --
          // a file can have perfectly valid JSON syntax but a corrupt/
          // malformed weight architecture that only throws once nam::get_dsp
          // actually tries to build the DSP graph from it. Confirmed on real
          // hardware: one such file caused occasional full-device reboots
          // when a switch happened to land on it, even though preload's
          // JSON-parse-only check let it through. Discarded immediately --
          // this is purely a validity check, the real DSP gets built fresh
          // per channel at actual switch time (see switch_model_in_background).
          // A file that fails this never enters g_cached_models at all, so
          // with N *.nam files and one corrupt, the pedal behaves as if there
          // are only N-1 -- exactly like the corrupt file was never there.
          { auto discard = nam::get_dsp(j); }
          found.push_back(CachedModel{filename, std::move(j)});
        }
        catch (...)
        {
          // Bad/corrupt file -- skip it, don't let one bad file block every
          // other model from loading.
        }
      }
      if (!found.empty())
        break; // Stop at first directory that has models
    }

    std::sort(found.begin(), found.end(),
              [](const CachedModel& a, const CachedModel& b) { return a.display_name < b.display_name; });

    g_calibration_cache.assign(found.size(), CalibrationEntry{});
    g_cached_models = std::move(found);
    if (!g_cached_models.empty())
      g_models_ready.store(true, std::memory_order_release);
    else
      g_preload_started.store(false, std::memory_order_release);
  });
}

// Fixed 101-position knob (0%..100%, one .nam file per whole percent) instead
// of the old scheme of dividing the sweep into N equal zones sized to however
// many files were found. With N files present, only steps 0..N-1 (the first
// N% of the sweep) select a file; steps N..100 are deliberately unmapped and
// must render as silence (see nam_process_gonk's fade_state==0 branch and its
// fade-to-silence swap) rather than falling back to the previous/nearest
// file -- that would silently hide the fact that no file is assigned there.
constexpr int kKnobSteps = 101;

// Returns:
//   >= 0  -- file index to load (equal to the step number itself: step N
//            plays g_cached_models[N]).
//   -1    -- valid, resolved position, but this step has no file assigned
//            (step number >= file_count) -- caller must produce silence.
//   -2    -- no *.nam files available at all -- caller should stay in the
//            pre-ready dry-passthrough state, same as before this scheme.
struct Biquad
{
  float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
  float a1 = 0.0f, a2 = 0.0f;
  float z1 = 0.0f, z2 = 0.0f;

  void reset()
  {
    z1 = 0.0f;
    z2 = 0.0f;
  }

  inline float process(float in)
  {
    float out = in * b0 + z1;
    z1 = in * b1 - out * a1 + z2;
    z2 = in * b2 - out * a2;
    return out;
  }

  void set_low_shelf(float sample_rate, float cutoff_hz, float gain_db)
  {
    if (std::abs(gain_db) < 0.05f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f;
      a1 = 0.0f; a2 = 0.0f;
      return;
    }
    const float A = std::pow(10.0f, gain_db / 40.0f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * cutoff_hz / sample_rate;
    const float cos_w = std::cos(w0);
    const float sin_w = std::sin(w0);
    const float alpha = sin_w / 2.0f * std::sqrt(2.0f);
    const float two_sqrt_A_alpha = 2.0f * std::sqrt(A) * alpha;

    const float a0 = (A + 1.0f) + (A - 1.0f) * cos_w + two_sqrt_A_alpha;
    b0 = (A * ((A + 1.0f) - (A - 1.0f) * cos_w + two_sqrt_A_alpha)) / a0;
    b1 = (2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_w)) / a0;
    b2 = (A * ((A + 1.0f) - (A - 1.0f) * cos_w - two_sqrt_A_alpha)) / a0;
    a1 = (-2.0f * ((A - 1.0f) + (A + 1.0f) * cos_w)) / a0;
    a2 = ((A + 1.0f) + (A - 1.0f) * cos_w - two_sqrt_A_alpha) / a0;
  }

  void set_high_shelf(float sample_rate, float cutoff_hz, float gain_db)
  {
    if (std::abs(gain_db) < 0.05f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f;
      a1 = 0.0f; a2 = 0.0f;
      return;
    }
    const float A = std::pow(10.0f, gain_db / 40.0f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * cutoff_hz / sample_rate;
    const float cos_w = std::cos(w0);
    const float sin_w = std::sin(w0);
    const float alpha = sin_w / 2.0f * std::sqrt(2.0f);
    const float two_sqrt_A_alpha = 2.0f * std::sqrt(A) * alpha;

    const float a0 = (A + 1.0f) - (A - 1.0f) * cos_w + two_sqrt_A_alpha;
    b0 = (A * ((A + 1.0f) + (A - 1.0f) * cos_w + two_sqrt_A_alpha)) / a0;
    b1 = (-2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_w)) / a0;
    b2 = (A * ((A + 1.0f) + (A - 1.0f) * cos_w - two_sqrt_A_alpha)) / a0;
    a1 = (2.0f * ((A - 1.0f) - (A + 1.0f) * cos_w)) / a0;
    a2 = ((A + 1.0f) - (A - 1.0f) * cos_w - two_sqrt_A_alpha) / a0;
  }

  void set_peaking(float sample_rate, float center_hz, float gain_db, float Q = 1.0f)
  {
    if (std::abs(gain_db) < 0.05f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f;
      a1 = 0.0f; a2 = 0.0f;
      return;
    }
    const float A = std::pow(10.0f, gain_db / 40.0f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * center_hz / sample_rate;
    const float cos_w = std::cos(w0);
    const float sin_w = std::sin(w0);
    const float alpha = sin_w / (2.0f * Q);

    const float a0 = 1.0f + alpha / A;
    b0 = (1.0f + alpha * A) / a0;
    b1 = (-2.0f * cos_w) / a0;
    b2 = (1.0f - alpha * A) / a0;
    a1 = (-2.0f * cos_w) / a0;
    a2 = (1.0f - alpha / A) / a0;
  }

  void set_high_pass(float sample_rate, float cutoff_hz, float Q = 0.707f)
  {
    const float w0 = 2.0f * static_cast<float>(M_PI) * cutoff_hz / sample_rate;
    const float cos_w = std::cos(w0);
    const float sin_w = std::sin(w0);
    const float alpha = sin_w / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    b0 = ((1.0f + cos_w) / 2.0f) / a0;
    b1 = (-(1.0f + cos_w)) / a0;
    b2 = ((1.0f + cos_w) / 2.0f) / a0;
    a1 = (-2.0f * cos_w) / a0;
    a2 = (1.0f - alpha) / a0;
  }

  void set_low_pass(float sample_rate, float cutoff_hz, float Q = 0.707f)
  {
    const float w0 = 2.0f * static_cast<float>(M_PI) * cutoff_hz / sample_rate;
    const float cos_w = std::cos(w0);
    const float sin_w = std::sin(w0);
    const float alpha = sin_w / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    b0 = ((1.0f - cos_w) / 2.0f) / a0;
    b1 = (1.0f - cos_w) / a0;
    b2 = ((1.0f - cos_w) / 2.0f) / a0;
    a1 = (-2.0f * cos_w) / a0;
    a2 = (1.0f - alpha) / a0;
  }
};

// ============================================================================
// Shared Physically-Grounded DSP Primitives (48 kHz base / 96 kHz oversampled)
// ============================================================================
static constexpr float kDspBaseFs = 48000.0f;
static constexpr float kDspOsFs = 96000.0f;

static inline float alpha_from_sec(float tau_sec, float fs)
{
  return (tau_sec <= 0.0f) ? 1.0f : (1.0f - std::exp(-1.0f / (tau_sec * fs)));
}

static inline float asym_tanh(float x, float bias)
{
  return std::tanh(x + bias) - std::tanh(bias);
}

static inline float triode_stage(float x, float drive, float asym, float& env,
                                 float atk_alpha, float rel_alpha)
{
  const float ax = std::fabs(x);
  env = (ax > env) ? (atk_alpha * ax + (1.0f - atk_alpha) * env)
                   : (rel_alpha * ax + (1.0f - rel_alpha) * env);
  const float bias = asym + 0.14f * std::min(env, 1.2f);
  return asym_tanh(drive * x, bias);
}

// Block output VOLUME taper (0..1): 0% = true mute (0.0f), 50% = unity_trim (nominal
// level matched across all custom amps), 100% = +8 dB clean reserve.
static inline float amp_block_volume_taper(float level_raw, float unity_trim)
{
  const float v = std::clamp(level_raw, 0.0f, 1.0f);
  if (v <= 0.0f)
    return 0.0f;
  const float norm = (v <= 0.50f)
                         ? std::pow(v / 0.50f, 1.55f)
                         : std::pow(10.0f, ((v - 0.50f) * 2.0f * 8.0f) / 20.0f);
  return norm * unity_trim;
}

// Literal amp MAST VOL (0..1 -> power-stage input drive):
//   0%      = 0.0f (cuts signal feeding the fixed 100% power stage -> silence)
//   0..70%  = audio-taper clean headroom up to the tube knee
//   70..100%= progressive power-tube compression, sag & 2f0 push without brutal fuzz
static inline float amp_master_drive(float master_raw, float drive_at_half, float drive_at_full)
{
  const float m = std::clamp(master_raw, 0.0f, 1.0f);
  if (m <= 0.0f)
    return 0.0f;
  if (m <= 0.50f)
    return drive_at_half * std::pow(m / 0.50f, 1.35f);
  const float t = (m - 0.50f) / 0.50f;
  const float curve = 0.75f * t + 0.25f * t * t;
  return drive_at_half + (drive_at_full - drive_at_half) * curve;
}

struct Filter2
{
  float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  float z1 = 0.0f, z2 = 0.0f;

  void reset() { z1 = 0.0f; z2 = 0.0f; }

  inline float tick(float x)
  {
    const float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }

  void set_hpf(float fs, float fc, float Q = 0.707f)
  {
    fc = std::clamp(fc, 20.0f, fs * 0.45f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float c = std::cos(w0), s = std::sin(w0);
    const float alpha = s / (2.0f * Q);
    const float a0 = 1.0f + alpha;
    b0 = ((1.0f + c) * 0.5f) / a0;
    b1 = (-(1.0f + c)) / a0;
    b2 = ((1.0f + c) * 0.5f) / a0;
    a1 = (-2.0f * c) / a0;
    a2 = (1.0f - alpha) / a0;
  }

  void set_lpf(float fs, float fc, float Q = 0.707f)
  {
    fc = std::clamp(fc, 20.0f, fs * 0.45f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float c = std::cos(w0), s = std::sin(w0);
    const float alpha = s / (2.0f * Q);
    const float a0 = 1.0f + alpha;
    b0 = ((1.0f - c) * 0.5f) / a0;
    b1 = (1.0f - c) / a0;
    b2 = ((1.0f - c) * 0.5f) / a0;
    a1 = (-2.0f * c) / a0;
    a2 = (1.0f - alpha) / a0;
  }

  void set_peaking(float fs, float fc, float Q, float gain_db)
  {
    if (std::fabs(gain_db) < 0.03f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f; a1 = 0.0f; a2 = 0.0f;
      return;
    }
    fc = std::clamp(fc, 20.0f, fs * 0.45f);
    const float A = std::pow(10.0f, gain_db * 0.025f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float c = std::cos(w0), s = std::sin(w0);
    const float alpha = s / (2.0f * Q);
    const float a0 = 1.0f + alpha / A;
    b0 = (1.0f + alpha * A) / a0;
    b1 = (-2.0f * c) / a0;
    b2 = (1.0f - alpha * A) / a0;
    a1 = (-2.0f * c) / a0;
    a2 = (1.0f - alpha / A) / a0;
  }

  void set_low_shelf(float fs, float fc, float gain_db, float S = 0.75f)
  {
    if (std::fabs(gain_db) < 0.03f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f; a1 = 0.0f; a2 = 0.0f;
      return;
    }
    fc = std::clamp(fc, 20.0f, fs * 0.45f);
    const float A = std::pow(10.0f, gain_db * 0.025f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float c = std::cos(w0), s = std::sin(w0);
    const float alpha = s * 0.5f * std::sqrt(std::max(1e-4f, (A + 1.0f / A) * (1.0f / S - 1.0f) + 2.0f));
    const float two_sqA_alpha = 2.0f * std::sqrt(A) * alpha;
    const float a0 = (A + 1.0f) + (A - 1.0f) * c + two_sqA_alpha;
    b0 = (A * ((A + 1.0f) - (A - 1.0f) * c + two_sqA_alpha)) / a0;
    b1 = (2.0f * A * ((A - 1.0f) - (A + 1.0f) * c)) / a0;
    b2 = (A * ((A + 1.0f) - (A - 1.0f) * c - two_sqA_alpha)) / a0;
    a1 = (-2.0f * ((A - 1.0f) + (A + 1.0f) * c)) / a0;
    a2 = ((A + 1.0f) + (A - 1.0f) * c - two_sqA_alpha) / a0;
  }

  void set_high_shelf(float fs, float fc, float gain_db, float S = 0.75f)
  {
    if (std::fabs(gain_db) < 0.03f)
    {
      b0 = 1.0f; b1 = 0.0f; b2 = 0.0f; a1 = 0.0f; a2 = 0.0f;
      return;
    }
    fc = std::clamp(fc, 20.0f, fs * 0.45f);
    const float A = std::pow(10.0f, gain_db * 0.025f);
    const float w0 = 2.0f * static_cast<float>(M_PI) * fc / fs;
    const float c = std::cos(w0), s = std::sin(w0);
    const float alpha = s * 0.5f * std::sqrt(std::max(1e-4f, (A + 1.0f / A) * (1.0f / S - 1.0f) + 2.0f));
    const float two_sqA_alpha = 2.0f * std::sqrt(A) * alpha;
    const float a0 = (A + 1.0f) - (A - 1.0f) * c + two_sqA_alpha;
    b0 = (A * ((A + 1.0f) + (A - 1.0f) * c + two_sqA_alpha)) / a0;
    b1 = (-2.0f * A * ((A - 1.0f) + (A + 1.0f) * c)) / a0;
    b2 = (A * ((A + 1.0f) + (A - 1.0f) * c - two_sqA_alpha)) / a0;
    a1 = (2.0f * ((A - 1.0f) - (A + 1.0f) * c)) / a0;
    a2 = ((A + 1.0f) - (A - 1.0f) * c - two_sqA_alpha) / a0;
  }
};

struct OnePoleLPF
{
  float a = 1.0f;
  float z = 0.0f;

  void reset() { z = 0.0f; }

  void set_cutoff(float fs, float fc)
  {
    fc = std::clamp(fc, 20.0f, fs * 0.48f);
    a = 1.0f - std::exp(-2.0f * static_cast<float>(M_PI) * fc / fs);
  }

  inline float tick(float x)
  {
    z += a * (x - z);
    return z;
  }
};

// Zero-droop 2x oversampler (zero-stuff + 4th-order Butterworth @ 21.2 kHz) +
// 4th-order Butterworth anti-aliasing downsampler. Flat within <0.25 dB up to 17 kHz.
struct Oversampler2x
{
  Filter2 up1, up2;
  Filter2 dn1, dn2;
  bool configured = false;

  void init()
  {
    // 4th-order Butterworth cascade @ 22.4 kHz (96 kHz fs -> <-0.25 dB @ 16 kHz roundtrip)
    up1.set_lpf(kDspOsFs, 22400.0f, 0.5412f);
    up2.set_lpf(kDspOsFs, 22400.0f, 1.3066f);
    dn1.set_lpf(kDspOsFs, 22400.0f, 0.5412f);
    dn2.set_lpf(kDspOsFs, 22400.0f, 1.3066f);
    configured = true;
  }

  void reset()
  {
    up1.reset();
    up2.reset();
    dn1.reset();
    dn2.reset();
    if (!configured)
      init();
  }

  inline void upsample(float x, float out[2])
  {
    if (!configured)
      init();
    out[0] = up2.tick(up1.tick(2.0f * x));
    out[1] = up2.tick(up1.tick(0.0f));
  }

  inline float downsample_tick(float os_sample)
  {
    return dn2.tick(dn1.tick(os_sample));
  }
};

struct ToneStackSpec
{
  float bass_fc = 120.0f;
  float bass_span_db = 24.0f;
  float bass_bias_db = 0.0f;

  float mid_fc = 650.0f;
  float mid_q = 0.72f;
  float mid_span_db = 22.0f;
  float mid_bias_db = 0.0f;

  float treble_fc = 1680.0f;
  float treble_span_db = 24.0f;
  float treble_bias_db = 0.0f;

  float pres_fc = 4150.0f;
  float pres_span_db = 22.0f;
  float pres_bias_db = 0.0f;
};

// Decoupled, high-authority 4-band amp tone stack:
// Bass (low shelf ~120 Hz), Mid (wide bell ~500-820 Hz),
// Treble (high shelf ~1.65-1.8 kHz), NFB Presence (high shelf ~4.1-4.3 kHz up to 20 kHz).
struct GuitarToneStack4
{
  Filter2 bass_f, mid_f, treble_f, pres_f;
  float last_b = -1.0f, last_m = -1.0f, last_t = -1.0f, last_p = -1.0f;

  void reset()
  {
    bass_f.reset();
    mid_f.reset();
    treble_f.reset();
    pres_f.reset();
    invalidate();
  }

  void invalidate()
  {
    last_b = -1.0f;
    last_m = -1.0f;
    last_t = -1.0f;
    last_p = -1.0f;
  }

  void update(const ToneStackSpec& spec, float bass, float mid, float treble, float presence,
              bool force = false)
  {
    if (force || std::fabs(bass - last_b) > 1e-4f)
    {
      bass_f.set_low_shelf(kDspBaseFs, spec.bass_fc,
                           spec.bass_bias_db + (bass - 0.5f) * spec.bass_span_db, 0.72f);
      last_b = bass;
    }
    if (force || std::fabs(mid - last_m) > 1e-4f)
    {
      mid_f.set_peaking(kDspBaseFs, spec.mid_fc, spec.mid_q,
                        spec.mid_bias_db + (mid - 0.5f) * spec.mid_span_db);
      last_m = mid;
    }
    if (force || std::fabs(treble - last_t) > 1e-4f)
    {
      treble_f.set_high_shelf(kDspBaseFs, spec.treble_fc,
                              spec.treble_bias_db + (treble - 0.5f) * spec.treble_span_db, 0.70f);
      last_t = treble;
    }
    if (force || std::fabs(presence - last_p) > 1e-4f)
    {
      pres_f.set_high_shelf(kDspBaseFs, spec.pres_fc,
                            spec.pres_bias_db + (presence - 0.5f) * spec.pres_span_db, 0.72f);
      last_p = presence;
    }
  }

  inline float tick(float x)
  {
    float y = bass_f.tick(x);
    y = mid_f.tick(y);
    y = treble_f.tick(y);
    return pres_f.tick(y);
  }
};

// Studio optical/VCA sustain engine with dual detectors:
// - Hold detector follows guitar decay at ~140-180 ms (ripple-free down to E2=82 Hz yet zero lag behind note fade)
// - Fast gate detector drops boost back to 1.0x within ~75 ms on palm stops / silence (< -60 dBFS)
struct ProgramSustain
{
  float env_hold = 0.0f;
  float env_fast = 0.0f;
  float gain_smooth = 1.0f;

  void reset()
  {
    env_hold = 0.0f;
    env_fast = 0.0f;
    gain_smooth = 1.0f;
  }

  inline float tick_core(float x, float atk_sec, float rel_sec, float target_rms,
                         float knee, float max_up_gain, float min_down_gain,
                         float comp_exp, float wet_mix)
  {
    const float ax = std::fabs(x);
    const float atk_a = alpha_from_sec(atk_sec, kDspBaseFs);
    const float rel_a = alpha_from_sec(rel_sec, kDspBaseFs);
    env_hold = (ax > env_hold) ? (atk_a * ax + (1.0f - atk_a) * env_hold)
                               : (rel_a * ax + (1.0f - rel_a) * env_hold);

    const float fast_atk = alpha_from_sec(0.002f, kDspBaseFs);
    const float fast_rel = alpha_from_sec(0.025f, kDspBaseFs);
    env_fast = (ax > env_fast) ? (fast_atk * ax + (1.0f - fast_atk) * env_fast)
                               : (fast_rel * ax + (1.0f - fast_rel) * env_fast);

    // Smooth noise-floor gate: upward boost engages above -63 dB (0.0007) and opens fully by -52 dB (0.0025)
    constexpr float kNoiseFloor = 0.0007f;
    constexpr float kNoiseSpan = 0.0018f;
    const float t_gate = std::clamp((env_fast - kNoiseFloor) / kNoiseSpan, 0.0f, 1.0f);
    const float smooth_gate = t_gate * t_gate * (3.0f - 2.0f * t_gate);

    const float ratio_term = target_rms / (env_hold + knee);
    const float leveled = (comp_exp >= 0.999f) ? ratio_term : std::pow(std::max(ratio_term, 1e-6f), comp_exp);
    const float raw_gain = std::clamp(leveled, min_down_gain, max_up_gain);
    const float gated_gain = (raw_gain > 1.0f) ? (1.0f + smooth_gate * (raw_gain - 1.0f)) : raw_gain;

    const float gs_alpha = alpha_from_sec(0.010f, kDspBaseFs);
    gain_smooth += gs_alpha * (gated_gain - gain_smooth);
    const float mix = std::clamp(wet_mix, 0.0f, 1.0f);
    return x * ((1.0f - mix) + mix * gain_smooth);
  }

  inline float tick(float x, float atk_sec, float rel_sec, float target_rms,
                    float knee, float max_up_gain, float min_down_gain = 0.25f)
  {
    return tick_core(x, atk_sec, rel_sec, target_rms, knee, max_up_gain, min_down_gain, 1.0f, 1.0f);
  }

  // Gentle finite-ratio parallel optical sustain: preserves pick transients and wide playing dynamics.
  inline float tick_optical(float x, float atk_sec, float rel_sec, float ref_level,
                            float knee, float ratio, float max_up_db, float max_down_db,
                            float wet_mix)
  {
    const float comp_exp = (ratio > 1.0f) ? (1.0f - 1.0f / ratio) : 0.0f;
    const float max_up_gain = std::pow(10.0f, std::max(0.0f, max_up_db) / 20.0f);
    const float min_down_gain = std::pow(10.0f, -std::max(0.0f, max_down_db) / 20.0f);
    return tick_core(x, atk_sec, rel_sec, ref_level, knee, max_up_gain, min_down_gain, comp_exp, wet_mix);
  }
};

// Mesa/Boogie Triple Crown (TC-100) Circuit-Modeled Tube Amplifier DSP:
// CH1 Clean/Drive, CH2 Lo-Gain British EL34 Crunch, CH3 Hi-Gain Tight Lead.
// Full 0..18 kHz amp-block bandwidth (zero speaker cab roll-off inside amp block).
struct MesaTripleCrownDSP
{
  using Filter2 = ::Filter2;

  static constexpr float kBaseFs = kDspBaseFs;
  static constexpr float kOsFs = kDspOsFs;

  Oversampler2x os2x;
  Filter2 in_hpf;
  Filter2 tight_peak;
  Filter2 inter1_hpf, inter2_hpf;

  GuitarToneStack4 tone_stack;
  Filter2 kerrang_peak, depth_thump, sub_tight_hpf, def_peak;

  float v1_env = 0.0f;
  float v2_env = 0.0f;
  float sag_env = 0.0f;
  float pwr_comp_env = 0.0f;
  float out_comp_env = 0.0f;

  int last_channel = -1;
  int last_bright = -1;
  float last_bass_tight = -1.0f;

  void reset()
  {
    os2x.reset();
    in_hpf.reset();
    tight_peak.reset();
    inter1_hpf.reset();
    inter2_hpf.reset();
    tone_stack.reset();
    kerrang_peak.reset();
    depth_thump.reset();
    sub_tight_hpf.reset();
    def_peak.reset();
    v1_env = 0.0f;
    v2_env = 0.0f;
    sag_env = 0.0f;
    pwr_comp_env = 0.0f;
    out_comp_env = 0.0f;
    last_channel = -1;
    last_bright = -1;
    last_bass_tight = -1.0f;
  }

  static inline float asym_tanh(float x, float bias) { return ::asym_tanh(x, bias); }

  static inline float triode_stage(float x, float drive, float asym, float& env,
                                   float atk = 0.04f, float rel = 0.0002f)
  {
    return ::triode_stage(x, drive, asym, env, atk, rel);
  }

  void update_filters(int channel, bool bright, float bass, float mid, float treble, float presence)
  {
    const int br = bright ? 1 : 0;
    const bool mode_changed = (channel != last_channel) || (br != last_bright);
    if (last_channel == -1)
    {
      reset();
      sub_tight_hpf.set_hpf(kBaseFs, 48.0f, 0.707f);
      depth_thump.set_peaking(kBaseFs, 96.0f, 1.15f, 1.5f);
    }

    if (mode_changed || (channel == 3 && std::fabs(bass - last_bass_tight) > 1e-4f))
    {
      const float ch3_tight_shift = (channel == 3) ? (0.5f - bass) * 28.0f : 0.0f;
      if (channel == 1)
      {
        in_hpf.set_hpf(kOsFs, bright ? 82.0f : 58.0f, 0.707f);
        tight_peak.set_peaking(kOsFs, 3000.0f, 0.85f, bright ? 2.5f : 0.8f);
        inter1_hpf.set_hpf(kOsFs, 65.0f, 0.707f);
        inter2_hpf.set_hpf(kOsFs, 70.0f, 0.707f);
        kerrang_peak.set_peaking(kBaseFs, 1600.0f, 1.0f, 0.0f);
        def_peak.set_peaking(kBaseFs, 2800.0f, 0.90f, 1.2f);
      }
      else if (channel == 2)
      {
        in_hpf.set_hpf(kOsFs, bright ? 110.0f : 72.0f, 0.707f);
        tight_peak.set_peaking(kOsFs, 2000.0f, 0.95f, bright ? 3.2f : 1.5f);
        inter1_hpf.set_hpf(kOsFs, bright ? 105.0f : 75.0f, 0.707f);
        inter2_hpf.set_hpf(kOsFs, 78.0f, 0.707f);
        kerrang_peak.set_peaking(kBaseFs, 1650.0f, 0.95f, 1.8f);
        def_peak.set_peaking(kBaseFs, 2900.0f, 0.90f, 1.4f);
      }
      else
      {
        in_hpf.set_hpf(kOsFs, std::clamp((bright ? 142.0f : 118.0f) + ch3_tight_shift, 90.0f, 168.0f), 0.707f);
        tight_peak.set_peaking(kOsFs, 1150.0f, 1.20f, bright ? 3.2f : 2.0f);
        inter1_hpf.set_hpf(kOsFs, bright ? 125.0f : 95.0f, 0.707f);
        inter2_hpf.set_hpf(kOsFs, 95.0f, 0.707f);
        kerrang_peak.set_peaking(kBaseFs, 1620.0f, 1.00f, 3.0f);
        def_peak.set_peaking(kBaseFs, 3000.0f, 0.90f, 1.6f);
      }
      last_channel = channel;
      last_bright = br;
      last_bass_tight = bass;
    }

    ToneStackSpec spec{};
    spec.bass_fc = 122.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = (channel == 3) ? 560.0f : 720.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 24.0f;
    spec.mid_bias_db = (channel == 3) ? -2.6f : (channel == 2 ? 0.8f : 0.0f);
    spec.treble_fc = 1650.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4150.0f;
    spec.pres_span_db = 22.0f;
    tone_stack.update(spec, bass, mid, treble, presence, mode_changed);
  }

  void process(const float* in, float* out, int32_t numFrames, int channel,
               float gain, float master, float level, float bass, float mid, float treble,
               float presence, bool bright)
  {
    update_filters(channel, bright, bass, mid, treble, presence);

    const float master_drive = amp_master_drive(master, 0.65f, 1.48f);
    const float ch_out_trim = (channel == 1) ? 0.88f : (channel == 2) ? 0.64f : 0.78f;
    const float out_level = amp_block_volume_taper(level, ch_out_trim);

    const float kV1Atk = alpha_from_sec(0.0005f, kOsFs);
    const float kV1Rel = alpha_from_sec(0.110f, kOsFs);
    const float kSagAtk = alpha_from_sec((channel == 3) ? 0.0045f : 0.0075f, kOsFs);
    const float kSagRel = alpha_from_sec(0.120f, kOsFs);
    const float kPwrAtk = alpha_from_sec(0.012f, kOsFs);
    const float kPwrRel = alpha_from_sec(0.150f, kOsFs);
    const float kOutCompAtk = alpha_from_sec(0.004f, kBaseFs);
    const float kOutCompRel = alpha_from_sec(0.095f, kBaseFs);

    for (int32_t i = 0; i < numFrames; ++i)
    {
      const float x_cur = std::clamp(in[i], -8.0f, 8.0f);
      float x_os[2];
      os2x.upsample(x_cur, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float sig = tight_peak.tick(in_hpf.tick(x_os[s]));

        if (channel == 1)
        {
          const float g1 = bright ? (1.35f + 4.2f * gain) : (0.85f + 2.0f * gain);
          const float g2 = bright ? (1.25f + 2.4f * gain) : (0.95f + 1.1f * gain);
          sig = triode_stage(sig, g1, 0.05f, v1_env, kV1Atk, kV1Rel);
          sig = inter1_hpf.tick(sig);
          sig = triode_stage(sig, g2, 0.07f, v2_env, kV1Atk, kV1Rel);
        }
        else if (channel == 2)
        {
          const float g1 = (0.75f + 6.80f * gain) * (bright ? 1.25f : 1.0f);
          sig = triode_stage(sig, g1, 0.10f, v1_env, kV1Atk, kV1Rel);
          sig = inter1_hpf.tick(sig);
          sig = triode_stage(sig, 1.15f + 2.70f * gain, 0.16f, v2_env, kV1Atk, kV1Rel);
          sig = inter2_hpf.tick(sig);
          sig = asym_tanh((1.20f + 0.85f * gain) * sig, 0.08f);
        }
        else
        {
          const float g1 = (1.10f + 11.50f * gain) * (bright ? 1.28f : 1.0f);
          sig = triode_stage(sig, g1, 0.12f, v1_env, kV1Atk, kV1Rel);
          sig = inter1_hpf.tick(sig);
          sig = asym_tanh((1.65f + 1.85f * gain) * sig, 0.14f);
          sig = inter2_hpf.tick(sig);
          sig = triode_stage(sig, 1.50f + 3.30f * gain, 0.34f, v2_env, kV1Atk, kV1Rel);
          sig = asym_tanh(1.75f * sig, -0.06f);
        }

        const float p_abs = std::fabs(sig);
        const float sag_depth = (channel == 3) ? 0.18f : 0.36f;
        sag_env = (p_abs > sag_env) ? (kSagAtk * p_abs + (1.0f - kSagAtk) * sag_env)
                                    : (kSagRel * p_abs + (1.0f - kSagRel) * sag_env);
        pwr_comp_env = (p_abs > pwr_comp_env) ? (kPwrAtk * p_abs + (1.0f - kPwrAtk) * pwr_comp_env)
                                              : (kPwrRel * p_abs + (1.0f - kPwrRel) * pwr_comp_env);
        const float pre_comp = 1.0f / (1.0f + 0.24f * std::min(pwr_comp_env, 1.5f));
        const float rail = 1.0f / (1.0f + sag_depth * std::min(sag_env, 1.6f));
        const float pwr_bias = 0.035f / rail;
        float pwr = rail * asym_tanh((sig * pre_comp * master_drive) / rail, pwr_bias);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = sub_tight_hpf.tick(os_out);
      y = depth_thump.tick(y);
      y = kerrang_peak.tick(y);
      y = def_peak.tick(y);

      const float abs_y = std::fabs(y);
      out_comp_env = (abs_y > out_comp_env)
                         ? (kOutCompAtk * abs_y + (1.0f - kOutCompAtk) * out_comp_env)
                         : (kOutCompRel * abs_y + (1.0f - kOutCompRel) * out_comp_env);
      const float comp_thresh = (channel == 3) ? 0.48f : 0.50f;
      if (out_comp_env > comp_thresh)
      {
        const float over = out_comp_env - comp_thresh;
        const float ratio_k = (channel == 3) ? 3.2f : 2.0f;
        const float target_env = comp_thresh + over / (1.0f + ratio_k * std::min(1.0f, over * 2.5f));
        y *= target_env / (out_comp_env + 1e-6f);
      }
      if (channel == 3)
        y = std::tanh(y * 0.90f) / 0.90f;

      // Apply user 4-band tone stack AFTER power pushback so all controls sweep with full authority
      y = tone_stack.tick(y) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.25f, 1.25f);
    }
  }
};


// Mesa/Boogie Lone Star (Channel 1 Clean, 100W/50W 6L6GC) Circuit-Modeled DSP (no cab sim):
// 2 lively high-voltage 12AX7 stages (V1A -> clean Blackface/Texas body -> V1B) with authentic 2f0 harmonic glow,
// 3D pick sparkle (2.8 kHz chime + 5.8 kHz air, full 0..18 kHz bandwidth), and Silicon (OFF) vs 5U4GB (ON) sag.
struct MesaLoneStarDSP
{
  using Filter2 = ::Filter2;
  static constexpr float kBaseFs = kDspBaseFs;
  static constexpr float kOsFs = kDspOsFs;

  Oversampler2x os2x;
  Filter2 in_hpf;
  Filter2 ts_mid_body;
  Filter2 sparkle_bell;
  Filter2 air_shelf;
  GuitarToneStack4 tone_stack;
  Filter2 depth_thump;
  Filter2 sub_tight_hpf;

  float v1a_env = 0.0f;
  float v1b_env = 0.0f;
  float dc_est = 0.0f;
  float sag_env = 0.0f;
  float comp_env = 0.0f;
  bool initialized = false;

  void reset()
  {
    os2x.reset();
    in_hpf.reset();
    ts_mid_body.reset();
    sparkle_bell.reset();
    air_shelf.reset();
    tone_stack.reset();
    depth_thump.reset();
    sub_tight_hpf.reset();
    v1a_env = 0.0f;
    v1b_env = 0.0f;
    dc_est = 0.0f;
    sag_env = 0.0f;
    comp_env = 0.0f;
    initialized = false;
  }

  void update_filters(float bass, float mid, float treble, float presence)
  {
    if (!initialized)
    {
      reset();
      in_hpf.set_hpf(kOsFs, 50.0f, 0.707f);
      ts_mid_body.set_peaking(kOsFs, 380.0f, 0.95f, 2.0f);
      sparkle_bell.set_peaking(kOsFs, 2800.0f, 0.80f, 2.4f);
      air_shelf.set_high_shelf(kBaseFs, 5800.0f, 1.8f, 0.65f);
      sub_tight_hpf.set_hpf(kBaseFs, 52.0f, 0.707f);
      depth_thump.set_peaking(kBaseFs, 84.0f, 1.25f, 1.6f);
      initialized = true;
    }
    ToneStackSpec spec{};
    spec.bass_fc = 115.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 520.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 22.0f;
    spec.treble_fc = 1650.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4200.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, presence);
  }

  void process(const float* in, float* out, int32_t numFrames,
               float gain, float master, float level, float bass, float mid, float treble,
               float presence, bool rectifier_5u4gb)
  {
    update_filters(bass, mid, treble, presence);

    const float g_v1a = 0.78f + 2.45f * gain;
    const float top35 = std::max(0.0f, gain - 0.65f) / 0.35f;
    const float g_v1b = 0.85f + 1.35f * gain + 2.25f * top35;
    const float master_drive = amp_master_drive(master, 0.58f, 1.32f);
    const float out_level = amp_block_volume_taper(level, 1.18f);

    const float kV1Atk = alpha_from_sec(0.003f, kOsFs);
    const float kV1Rel = alpha_from_sec(0.085f, kOsFs);
    const float sag_depth = rectifier_5u4gb ? 0.36f : 0.10f;
    const float sag_atk = alpha_from_sec(rectifier_5u4gb ? 0.014f : 0.006f, kOsFs);
    const float sag_rel = alpha_from_sec(0.130f, kOsFs);

    for (int32_t i = 0; i < numFrames; ++i)
    {
      const float x_cur = std::clamp(in[i], -8.0f, 8.0f);
      float x_os[2];
      os2x.upsample(x_cur, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float sig = in_hpf.tick(x_os[s]);

        // V1A: lively Texas tube triode w/ genuine DC-blocked even-order 2f0 3D shimmer
        const float ax1 = std::fabs(sig);
        v1a_env = (ax1 > v1a_env) ? (kV1Atk * ax1 + (1.0f - kV1Atk) * v1a_env)
                                  : (kV1Rel * ax1 + (1.0f - kV1Rel) * v1a_env);
        float d1 = g_v1a * sig;
        const float d1_sq = (d1 * d1) / (1.0f + std::fabs(d1));
        dc_est = 0.9985f * dc_est + 0.0015f * d1_sq;
        // Negative quadratic aligns polarity with asym_tanh(+bias) so V1A/V1B/6L6 2f0 harmonics add constructively
        d1 = d1 - 0.12f * (d1_sq - dc_est);
        sig = asym_tanh(d1 * 0.34f, 0.055f + 0.05f * std::min(v1a_env, 1.0f)) / 0.34f;

        // Open Texas voice + sparkle (no mid mud, no 3.5 kHz cut)
        sig = sparkle_bell.tick(ts_mid_body.tick(sig));

        // V1B: responsive high-headroom clean-to-edge breakup stage
        const float ax2 = std::fabs(sig);
        v1b_env = (ax2 > v1b_env) ? (kV1Atk * ax2 + (1.0f - kV1Atk) * v1b_env)
                                  : (kV1Rel * ax2 + (1.0f - kV1Rel) * v1b_env);
        const float v1b_comp = 1.0f / (1.0f + 0.12f * std::min(v1b_env, 1.5f));
        sig = asym_tanh(g_v1b * v1b_comp * sig * 0.42f, 0.065f) / 0.42f;

        // 6L6GC Class AB Push-Pull power stage w/ musical bounce & clean headroom
        const float p_abs = std::fabs(sig);
        sag_env = (p_abs > sag_env) ? (sag_atk * p_abs + (1.0f - sag_atk) * sag_env)
                                    : (sag_rel * p_abs + (1.0f - sag_rel) * sag_env);
        comp_env = (p_abs > comp_env) ? (kV1Atk * p_abs + (1.0f - kV1Atk) * comp_env)
                                      : (sag_rel * p_abs + (1.0f - sag_rel) * comp_env);
        const float pwr_comp = 1.0f / (1.0f + 0.14f * std::min(comp_env, 1.5f));
        const float rail = 1.0f / (1.0f + sag_depth * std::min(sag_env, 1.6f));
        const float pwr_bias = 0.04f / rail;
        float pwr = rail * asym_tanh((sig * pwr_comp * master_drive) / rail, pwr_bias);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = sub_tight_hpf.tick(os_out);
      y = depth_thump.tick(y);
      y = air_shelf.tick(y);
      y = tone_stack.tick(y) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.5f, 1.5f);
    }
  }
};


// Dumble Steel String Singer (#002, 100W-150W 6L6GC/6550) Clean Headroom DSP (no cab sim):
// 2 linear 12AX7 stages w/ 0.1uF deep sub-bass extension (38 Hz HPF), 390 Hz scoop,
// 2.2 kHz piano-like chime, rigid silicon supply, and full 0..18 kHz bandwidth.
struct DumbleSSSDSP
{
  using Filter2 = ::Filter2;
  static constexpr float kBaseFs = kDspBaseFs;
  static constexpr float kOsFs = kDspOsFs;

  Oversampler2x os2x;
  Filter2 in_hpf;
  Filter2 mid_scoop;
  Filter2 chime_peak;
  Filter2 bright_shelf;
  GuitarToneStack4 tone_stack;
  Filter2 depth_thump;
  Filter2 air_peak;

  float sag_env = 0.0f;
  float pi_comp_env = 0.0f;
  bool initialized = false;
  int last_bright = -1;

  void reset()
  {
    os2x.reset();
    in_hpf.reset();
    mid_scoop.reset();
    chime_peak.reset();
    bright_shelf.reset();
    tone_stack.reset();
    depth_thump.reset();
    air_peak.reset();
    sag_env = 0.0f;
    pi_comp_env = 0.0f;
    initialized = false;
    last_bright = -1;
  }

  void update_filters(bool bright, float bass, float mid, float treble, float presence)
  {
    if (!initialized)
    {
      reset();
      in_hpf.set_hpf(kOsFs, 35.0f, 0.707f);
      mid_scoop.set_peaking(kOsFs, 400.0f, 0.85f, -4.2f);
      chime_peak.set_peaking(kOsFs, 2200.0f, 0.90f, 3.0f);
      depth_thump.set_peaking(kBaseFs, 90.0f, 1.05f, 1.6f);
      air_peak.set_peaking(kBaseFs, 4800.0f, 0.85f, 1.5f);
      initialized = true;
    }
    const int br = bright ? 1 : 0;
    if (br != last_bright)
    {
      bright_shelf.set_high_shelf(kOsFs, 3800.0f, bright ? 3.5f : 0.0f, 0.72f);
      last_bright = br;
    }
    ToneStackSpec spec{};
    spec.bass_fc = 122.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 500.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 22.0f;
    spec.treble_fc = 1700.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4200.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, presence);
  }

  void process(const float* in, float* out, int32_t numFrames,
               float gain, float master, float level, float bass, float mid, float treble,
               float presence, bool bright)
  {
    update_filters(bright, bass, mid, treble, presence);

    const float pre_gain = 0.50f + 1.85f * gain;
    const float master_drive = amp_master_drive(master, 0.50f, 1.18f);
    const float out_level = amp_block_volume_taper(level, 3.45f);

    const float kSagAtk = alpha_from_sec(0.005f, kOsFs);
    const float kSagRel = alpha_from_sec(0.058f, kOsFs);
    const float kPiAtk = alpha_from_sec(0.013f, kOsFs);
    const float kPiRel = alpha_from_sec(0.130f, kOsFs);

    for (int32_t i = 0; i < numFrames; ++i)
    {
      const float x_cur = std::clamp(in[i], -8.0f, 8.0f);
      float x_os[2];
      os2x.upsample(x_cur, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float sig = in_hpf.tick(x_os[s]);
        sig = std::tanh(pre_gain * sig * 0.50f) / 0.50f;
        sig = bright_shelf.tick(chime_peak.tick(mid_scoop.tick(sig)));

        const float p_abs = std::fabs(sig);
        sag_env = (p_abs > sag_env) ? (kSagAtk * p_abs + (1.0f - kSagAtk) * sag_env)
                                    : (kSagRel * p_abs + (1.0f - kSagRel) * sag_env);
        pi_comp_env = (p_abs > pi_comp_env) ? (kPiAtk * p_abs + (1.0f - kPiAtk) * pi_comp_env)
                                            : (kPiRel * p_abs + (1.0f - kPiRel) * pi_comp_env);
        const float pi_comp = 1.0f / (1.0f + 0.16f * std::min(pi_comp_env, 1.5f));
        const float rail = 1.0f / (1.0f + 0.05f * std::min(sag_env, 1.5f));
        float pwr = rail * std::tanh((sig * pi_comp * master_drive) / rail);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = depth_thump.tick(os_out);
      y = tone_stack.tick(air_peak.tick(y)) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.5f, 1.5f);
    }
  }
};


// Dumble Overdrive Special (#102/#124, Overdrive Channel) Circuit-Modeled DSP (no cab sim):
// True +12.5 dB high-headroom FET Input (Bright=ON), Skyliner voicing, zero 3.8 kHz choke (0..18 kHz open),
// and raw "gnarly" reinforced-asymmetry grid-conduction OD1/OD2 growl + singing bloom compression.
struct DumbleODSDSP
{
  using Filter2 = ::Filter2;
  static constexpr float kBaseFs = kDspBaseFs;
  static constexpr float kOsFs = kDspOsFs;

  Oversampler2x os2x;
  Filter2 in_hpf;
  Filter2 skyliner_mid;
  Filter2 bright_shelf;
  Filter2 inter_hpf;
  Filter2 gnarly_growl;
  GuitarToneStack4 tone_stack;
  Filter2 depth_thump;
  Filter2 sub_tight_hpf;
  Filter2 def_peak;

  float od1_env = 0.0f;
  float od2_env = 0.0f;
  float sag_env = 0.0f;
  float pwr_comp_env = 0.0f;
  bool initialized = false;
  int last_bright = -1;

  void reset()
  {
    os2x.reset();
    in_hpf.reset();
    skyliner_mid.reset();
    bright_shelf.reset();
    inter_hpf.reset();
    gnarly_growl.reset();
    tone_stack.reset();
    depth_thump.reset();
    sub_tight_hpf.reset();
    def_peak.reset();
    od1_env = 0.0f;
    od2_env = 0.0f;
    sag_env = 0.0f;
    pwr_comp_env = 0.0f;
    initialized = false;
    last_bright = -1;
  }

  void update_filters(bool bright, float bass, float mid, float treble, float presence)
  {
    if (!initialized)
    {
      reset();
      in_hpf.set_hpf(kOsFs, 65.0f, 0.707f);
      skyliner_mid.set_peaking(kOsFs, 580.0f, 0.90f, 2.0f);
      inter_hpf.set_hpf(kOsFs, 75.0f, 0.707f);
      gnarly_growl.set_peaking(kOsFs, 1550.0f, 0.90f, 1.8f);
      sub_tight_hpf.set_hpf(kBaseFs, 50.0f, 0.707f);
      depth_thump.set_peaking(kBaseFs, 102.0f, 1.05f, 1.4f);
      def_peak.set_peaking(kBaseFs, 4800.0f, 0.85f, 1.8f);
      initialized = true;
    }
    const int br = bright ? 1 : 0;
    if (br != last_bright)
    {
      // FET Input adds crisp JFET bite (+3.8 dB @ 2.2 kHz) on top of +12.5 dB front-end push
      bright_shelf.set_peaking(kOsFs, 2200.0f, 0.80f, bright ? 3.8f : 0.0f);
      last_bright = br;
    }
    ToneStackSpec spec{};
    spec.bass_fc = 122.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 620.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 22.0f;
    spec.treble_fc = 1680.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4150.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, presence);
  }

  void process(const float* in, float* out, int32_t numFrames,
               float gain, float master, float level, float bass, float mid, float treble,
               float presence, bool bright)
  {
    update_filters(bright, bass, mid, treble, presence);

    // Calibrated so Normal (Bright=OFF) sings cleanly/creamy, and FET Input (Bright=ON) slams hard into gnarly OD
    const float g_clean = 0.75f + 1.55f * gain;
    const float g_od1 = 0.75f + 4.50f * gain;
    const float g_od2 = 0.80f + 2.90f * gain;
    const float master_drive = amp_master_drive(master, 0.62f, 1.45f);
    const float out_level = amp_block_volume_taper(level, 0.64f);

    const float kAtk = alpha_from_sec(0.030f, kOsFs);
    const float kRel = alpha_from_sec(0.210f, kOsFs);
    const float kSagAtk = alpha_from_sec(0.008f, kOsFs);
    const float kSagRel = alpha_from_sec(0.130f, kOsFs);

    for (int32_t i = 0; i < numFrames; ++i)
    {
      const float x_cur = std::clamp(in[i], -8.0f, 8.0f);
      float x_os[2];
      os2x.upsample(x_cur, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float sig = in_hpf.tick(x_os[s]);

        // True FET Input (Bright=ON): +12.5 dB (4.2x) high-rail JFET push + 2f0 snap without pre-squashing
        if (bright)
        {
          sig *= 4.2f;
          sig = sig + 0.065f * (sig * sig) / (1.0f + std::fabs(sig));
          sig = std::tanh(sig * 0.16f + 0.03f) / 0.16f - std::tanh(0.03f) / 0.16f;
        }

        sig = asym_tanh(g_clean * sig, 0.08f);
        sig = bright_shelf.tick(skyliner_mid.tick(sig));

        // Raw "gnarly" ODS cascade: aligned positive grid-conduction asymmetry + odd/even teeth (open 0..18 kHz)
        sig = inter_hpf.tick(sig);
        sig = triode_stage(sig, g_od1, 0.22f, od1_env, kAtk, kRel);
        sig = gnarly_growl.tick(sig);
        const float od2_in = sig + 0.09f * std::tanh(4.2f * sig);
        sig = triode_stage(od2_in, g_od2, 0.28f, od2_env, kAtk, kRel);

        const float p_abs = std::fabs(sig);
        sag_env = (p_abs > sag_env) ? (kSagAtk * p_abs + (1.0f - kSagAtk) * sag_env)
                                    : (kSagRel * p_abs + (1.0f - kSagRel) * sag_env);
        pwr_comp_env = (p_abs > pwr_comp_env) ? (kAtk * p_abs + (1.0f - kAtk) * pwr_comp_env)
                                              : (kRel * p_abs + (1.0f - kRel) * pwr_comp_env);
        const float pre_comp = 1.0f / (1.0f + 0.22f * std::min(pwr_comp_env, 1.5f));
        const float rail = 1.0f / (1.0f + 0.24f * std::min(sag_env, 1.6f));
        const float pwr_bias = 0.075f / rail;
        float pwr = rail * asym_tanh((sig * pre_comp * master_drive) / rail, pwr_bias);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = sub_tight_hpf.tick(os_out);
      y = depth_thump.tick(y);
      y = def_peak.tick(y);
      y = tone_stack.tick(y) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.5f, 1.5f);
    }
  }
};

// Elmwood Bonneville 50 (Boutique Swedish 50W EL34 Hybrid: Raw British/Swedish Plexi-AC30 Crunch + Drastic Sustain)
// Full 0..18 kHz open bandwidth, zero stacked low-pass cuts, responsive Tone Stack.
struct ElmwoodBonnevilleDSP
{
  using Filter2 = ::Filter2;

  Oversampler2x os2x;
  ProgramSustain sustain_engine;

  Filter2 pre_hpf;
  Filter2 pre_clarity;
  Filter2 bright_chime;
  Filter2 inter_hpf;

  Filter2 post_wood_body;   // Organic 880 Hz Swedish/British Plexi bark (+1.8 dB)
  Filter2 post_kerrang;     // 2.15 kHz open-pore EL34 teeth (+1.8 dB)
  GuitarToneStack4 tone_stack;

  float v1_env = 0.0f;
  float v2_env = 0.0f;
  float v3_env = 0.0f;
  float v4_env = 0.0f;
  float cf_env = 0.0f;
  float pwr_sag_env = 0.0f;

  bool initialized = false;
  int last_bright = -1;

  void reset(double /*sr*/ = 48000.0)
  {
    os2x.reset();
    sustain_engine.reset();
    pre_hpf.reset();
    pre_clarity.reset();
    bright_chime.reset();
    inter_hpf.reset();
    post_wood_body.reset();
    post_kerrang.reset();
    tone_stack.reset();
    v1_env = v2_env = v3_env = v4_env = cf_env = 0.0f;
    pwr_sag_env = 0.0f;
    last_bright = -1;
    initialized = false;
  }

  void update_controls(float bass, float mid, float treble, float pres, bool bright)
  {
    if (!initialized)
    {
      reset();
      pre_hpf.set_hpf(kDspOsFs, bright ? 95.0f : 75.0f, 0.707f);
      pre_clarity.set_peaking(kDspOsFs, 2200.0f, 1.00f, 2.0f);
      inter_hpf.set_hpf(kDspOsFs, 75.0f, 0.707f);
      post_wood_body.set_peaking(kDspBaseFs, 880.0f, 0.90f, 1.8f);
      post_kerrang.set_peaking(kDspBaseFs, 2150.0f, 0.95f, 1.8f);
      initialized = true;
    }

    const int br = bright ? 1 : 0;
    if (br != last_bright)
    {
      pre_hpf.set_hpf(kDspOsFs, bright ? 95.0f : 75.0f, 0.707f);
      bright_chime.set_peaking(kDspOsFs, 2600.0f, 0.85f, bright ? 3.2f : 0.0f);
      last_bright = br;
    }

    ToneStackSpec spec{};
    spec.bass_fc = 85.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 820.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 22.0f;
    spec.treble_fc = 1680.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 3650.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, pres);
  }

  void process(const float* in, float* out, int n,
               float gain_raw, float master_raw, float level_raw,
               float bass_raw, float mid_raw, float treble_raw, float pres_raw,
               bool bright_on)
  {
    const float gain = std::clamp(gain_raw, 0.0f, 1.0f);
    const float master = std::clamp(master_raw, 0.0f, 1.0f);
    const float level = std::clamp(level_raw, 0.0f, 1.0f);
    const float bass = std::clamp(bass_raw, 0.0f, 1.0f);
    const float mid = std::clamp(mid_raw, 0.0f, 1.0f);
    const float treble = std::clamp(treble_raw, 0.0f, 1.0f);
    const float pres = std::clamp(pres_raw, 0.0f, 1.0f);

    update_controls(bass, mid, treble, pres, bright_on);

    // Fixed triode & EL34 sag alphas at 96 kHz (alpha = 1 - exp)
    const float tri_atk = alpha_from_sec(0.004f, kDspOsFs);
    const float tri_rel = alpha_from_sec(0.180f, kDspOsFs);
    const float pwr_atk = alpha_from_sec(0.009f, kDspOsFs);
    const float pwr_rel = alpha_from_sec(0.110f, kDspOsFs);

    // Raw British/Swedish 4x12AX7 open-pore Plexi gain staging
    const float drive1 = 0.80f + gain * 5.80f;
    const float drive2 = 0.85f + gain * 4.20f;
    const float drive3 = 0.80f + gain * 3.50f;
    const float drive4 = 0.80f + gain * 2.50f;

    const float master_drive = amp_master_drive(master, 0.64f, 1.50f);
    const float out_level = amp_block_volume_taper(level, 0.54f);

    for (int i = 0; i < n; ++i)
    {
      float x = std::isfinite(in[i]) ? std::clamp(in[i], -8.0f, 8.0f) : 0.0f;

      // Drastic singing EL34 harmonic sustain engine (10 ms attack, 150 ms fast-tracking release, +20 dB gated recovery)
      x = sustain_engine.tick(x, 0.010f, 0.150f, 0.160f, 0.006f, 10.5f, 0.30f);

      float x_os[2];
      os2x.upsample(x, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float sig = bright_chime.tick(pre_clarity.tick(pre_hpf.tick(x_os[s])));

        // Coarse-grain asymmetric British Plexi cascade
        float s1 = triode_stage(sig, drive1, 0.18f, v1_env, tri_atk, tri_rel);
        s1 = inter_hpf.tick(s1);
        float s2 = triode_stage(s1, drive2, -0.14f, v2_env, tri_atk, tri_rel);
        float s3 = triode_stage(s2, drive3, 0.15f, v3_env, tri_atk, tri_rel);
        float s4 = triode_stage(s3, drive4, -0.10f, v4_env, tri_atk, tri_rel);
        float cf = triode_stage(s4, 0.90f, 0.06f, cf_env, tri_atk, tri_rel);

        // 50W EL34 Push-Pull power stage
        const float abs_ts = std::fabs(cf * master_drive);
        pwr_sag_env = (abs_ts > pwr_sag_env) ? (pwr_atk * abs_ts + (1.0f - pwr_atk) * pwr_sag_env)
                                             : (pwr_rel * abs_ts + (1.0f - pwr_rel) * pwr_sag_env);
        const float rail = 1.0f / (1.0f + 0.16f * std::min(1.5f, pwr_sag_env));
        float pwr = rail * asym_tanh((cf * master_drive) / rail, 0.035f / rail);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = post_kerrang.tick(post_wood_body.tick(os_out));
      y = tone_stack.tick(y) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.5f, 1.5f);
    }
  }
};

// IRENE ("Clean Reference Engine" - High-Voltage Hybrid: Dumble SSS speed/headroom + Mesa Lone Star 3D warmth)
// Studio optical clean sustain + pure 2f0 400V even-order halo + full 0..18 kHz open top end.
struct IreneCleanDSP
{
  using Filter2 = ::Filter2;

  Oversampler2x os2x;
  ProgramSustain sustain_engine;

  Filter2 pre_hpf;        // 68 Hz 2nd-order HPF (mix-ready fundamental clarity)
  Filter2 pre_low_body;   // 105 Hz (+1.0 dB) balanced piano low weight
  Filter2 pre_lowmid_dip; // 320 Hz (-2.0 dB) surgical anti-boxiness / mix space
  Filter2 pre_vocal_mid;  // 1050 Hz (+2.0 dB) articulate mix-friendly mids
  Filter2 pre_chime;      // 2.8 kHz (+3.2 dB) crystal bell string separation
  Filter2 bright_air;     // Studio air shelf @ 5.2 kHz (+2.6 dB base, +5.4 dB w/ Bright)
  GuitarToneStack4 tone_stack;

  float dc_est = 0.0f;
  float pwr_sag_env = 0.0f;
  bool initialized = false;
  int last_bright = -1;

  void reset(double /*sr*/ = 48000.0)
  {
    os2x.reset();
    sustain_engine.reset();
    pre_hpf.reset();
    pre_low_body.reset();
    pre_lowmid_dip.reset();
    pre_vocal_mid.reset();
    pre_chime.reset();
    bright_air.reset();
    tone_stack.reset();
    dc_est = 0.0f;
    pwr_sag_env = 0.0f;
    last_bright = -1;
    initialized = false;
  }

  void update_controls(float bass, float mid, float treble, float pres, bool bright)
  {
    if (!initialized)
    {
      reset();
      pre_hpf.set_hpf(kDspBaseFs, 68.0f, 0.707f);
      pre_low_body.set_low_shelf(kDspBaseFs, 105.0f, 1.0f, 0.80f);
      pre_lowmid_dip.set_peaking(kDspBaseFs, 320.0f, 0.95f, -2.0f);
      pre_vocal_mid.set_peaking(kDspBaseFs, 1050.0f, 0.85f, 2.0f);
      pre_chime.set_peaking(kDspBaseFs, 2800.0f, 1.05f, 3.2f);
      bright_air.set_high_shelf(kDspBaseFs, 5200.0f, bright ? 5.4f : 2.6f, 0.70f);
      initialized = true;
    }

    const int br = bright ? 1 : 0;
    if (br != last_bright)
    {
      bright_air.set_high_shelf(kDspBaseFs, 5200.0f, bright ? 5.4f : 2.6f, 0.70f);
      last_bright = br;
    }

    ToneStackSpec spec{};
    spec.bass_fc = 118.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 820.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 22.0f;
    spec.treble_fc = 1800.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4400.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, pres);
  }

  void process(const float* in, float* out, int n,
               float gain_raw, float master_raw, float level_raw,
               float bass_raw, float mid_raw, float treble_raw, float pres_raw,
               bool bright_on)
  {
    const float gain = std::clamp(gain_raw, 0.0f, 1.0f);
    const float master = std::clamp(master_raw, 0.0f, 1.0f);
    const float level = std::clamp(level_raw, 0.0f, 1.0f);
    const float bass = std::clamp(bass_raw, 0.0f, 1.0f);
    const float mid = std::clamp(mid_raw, 0.0f, 1.0f);
    const float treble = std::clamp(treble_raw, 0.0f, 1.0f);
    const float pres = std::clamp(pres_raw, 0.0f, 1.0f);

    update_controls(bass, mid, treble, pres, bright_on);

    const float pwr_atk = alpha_from_sec(0.012f, kDspOsFs);
    const float pwr_rel = alpha_from_sec(0.110f, kDspOsFs);

    const float pre_gain = 0.62f + gain * 2.10f;
    const float k2_even = 0.110f + gain * 0.120f;
    const float master_drive = amp_master_drive(master, 0.52f, 1.24f);
    const float out_level = amp_block_volume_taper(level, 2.75f);

    for (int i = 0; i < n; ++i)
    {
      float x = std::isfinite(in[i]) ? std::clamp(in[i], -8.0f, 8.0f) : 0.0f;

      // High-pass subsonic before leveling so bass thuds never duck sustain
      x = pre_hpf.tick(x);

      // Gentle studio optical sustain (35 ms attack preserves pick transient, 220 ms release, 1.75:1 ratio, +5/-4.5 dB cap, 55% parallel mix)
      x = sustain_engine.tick_optical(x, 0.035f, 0.220f, 0.085f, 0.0040f, 1.75f, 5.0f, 4.5f, 0.55f);

      // IRENE signature sculpting (full 0..18 kHz open air, ZERO 6 kHz silk cut)
      x = pre_low_body.tick(x);
      x = pre_lowmid_dip.tick(x);
      x = pre_vocal_mid.tick(x);
      x = pre_chime.tick(x);
      x = bright_air.tick(x);

      float x_os[2];
      os2x.upsample(x, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        // High-voltage 400V linear triode + pure 2f0 halo (1.5-2.5% 2f0, <0.1% 3f0)
        float sig = x_os[s] * pre_gain;
        const float sq = sig * sig;
        dc_est = 0.9985f * dc_est + 0.0015f * sq;
        sig = sig + k2_even * (sq - dc_est);
        sig = std::tanh(sig * 0.18f) / 0.18f;

        // 100W 6L6GC/KT66 power stage
        const float abs_p = std::fabs(sig * master_drive);
        pwr_sag_env = (abs_p > pwr_sag_env) ? (pwr_atk * abs_p + (1.0f - pwr_atk) * pwr_sag_env)
                                            : (pwr_rel * abs_p + (1.0f - pwr_rel) * pwr_sag_env);
        const float rail = 1.0f / (1.0f + 0.12f * std::min(1.5f, pwr_sag_env));
        float pwr = rail * asym_tanh((sig * master_drive) / rail, 0.015f);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = tone_stack.tick(os_out) * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.4f, 1.4f);
    }
  }
};

// JON ("Modern Progressive Overdrive Engine" - 3-Voice Progressive Cascade: Clean/Blues -> Liquid Lead -> Surgical Tight Metal)
// Distinct from Elmwood: modern fine-grain Soldano/Triple-Crown precision, zero 4.5/7.5 kHz LPF choke, unlocked Tone Stack.
struct JonOverdriveDSP
{
  using Filter2 = ::Filter2;

  Oversampler2x os2x;
  Filter2 pre_hpf;
  Filter2 pre_dip;
  Filter2 pre_focus;
  Filter2 bright_boost;

  Filter2 inter_hpf;
  Filter2 post_clean_dip;
  Filter2 post_soldano_edge;
  Filter2 nfb_punch;
  GuitarToneStack4 tone_stack;

  float v2_env = 0.0f;
  float v3_env = 0.0f;
  float v4_env = 0.0f;
  float cf_env = 0.0f;
  float pwr_sag_env = 0.0f;

  bool initialized = false;
  float last_gain_eq = -999.0f;
  float last_bass_tight = -999.0f;
  int last_bright = -1;

  void reset(double /*sr*/ = 48000.0)
  {
    os2x.reset();
    pre_hpf.reset();
    pre_dip.reset();
    pre_focus.reset();
    bright_boost.reset();
    inter_hpf.reset();
    post_clean_dip.reset();
    post_soldano_edge.reset();
    nfb_punch.reset();
    tone_stack.reset();
    v2_env = v3_env = v4_env = cf_env = 0.0f;
    pwr_sag_env = 0.0f;
    last_gain_eq = last_bass_tight = -999.0f;
    last_bright = -1;
    initialized = false;
  }

  void update_controls(float gain, float bass, float mid, float treble, float pres, bool bright)
  {
    if (!initialized)
    {
      reset();
      inter_hpf.set_hpf(kDspOsFs, 115.0f, 0.707f);
      post_clean_dip.set_peaking(kDspBaseFs, 340.0f, 0.90f, -2.2f);
      post_soldano_edge.set_peaking(kDspBaseFs, 1450.0f, 0.95f, 2.2f);
      nfb_punch.set_peaking(kDspBaseFs, 78.0f, 1.15f, 1.0f);
      initialized = true;
    }

    const int br = bright ? 1 : 0;
    if (br != last_bright)
    {
      bright_boost.set_peaking(kDspOsFs, 2300.0f, 0.90f, bright ? 3.2f : 0.0f);
      last_bright = br;
    }

    if (std::fabs(gain - last_gain_eq) > 0.008f || std::fabs(bass - last_bass_tight) > 0.01f)
    {
      const float tight_trim = (gain > 0.55f) ? (0.5f - bass) * 28.0f : 0.0f;
      const float base_hpf = (gain < 0.5f)
                                 ? (68.0f + (gain / 0.5f) * 37.0f)
                                 : (105.0f + ((gain - 0.5f) / 0.5f) * 32.0f);
      pre_hpf.set_hpf(kDspOsFs, std::clamp(base_hpf + tight_trim, 60.0f, 165.0f), 0.707f);

      const float dip_db = (gain > 0.45f) ? (-3.8f * std::min(1.0f, (gain - 0.45f) / 0.45f)) : 0.0f;
      pre_dip.set_peaking(kDspOsFs, 320.0f, 0.95f, dip_db);

      const float focus_fc = (gain < 0.5f)
                                 ? (520.0f + (gain / 0.5f) * 140.0f)
                                 : (660.0f + ((gain - 0.5f) / 0.5f) * 740.0f);
      const float focus_db = (gain < 0.5f)
                                 ? (1.2f + (gain / 0.5f) * 1.4f)
                                 : (2.6f - ((gain - 0.5f) / 0.5f) * 0.2f);
      pre_focus.set_peaking(kDspOsFs, focus_fc, 1.00f, focus_db);
      last_gain_eq = gain;
      last_bass_tight = bass;
    }

    ToneStackSpec spec{};
    spec.bass_fc = 122.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = 640.0f;
    spec.mid_q = 0.72f;
    spec.mid_span_db = 24.0f;
    spec.treble_fc = 1700.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = 4200.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, pres);
  }

  void process(const float* in, float* out, int n,
               float gain_raw, float master_raw, float level_raw,
               float bass_raw, float mid_raw, float treble_raw, float pres_raw,
               bool bright_on)
  {
    const float gain = std::clamp(gain_raw, 0.0f, 1.0f);
    const float master = std::clamp(master_raw, 0.0f, 1.0f);
    const float level = std::clamp(level_raw, 0.0f, 1.0f);
    const float bass = std::clamp(bass_raw, 0.0f, 1.0f);
    const float mid = std::clamp(mid_raw, 0.0f, 1.0f);
    const float treble = std::clamp(treble_raw, 0.0f, 1.0f);
    const float pres = std::clamp(pres_raw, 0.0f, 1.0f);

    update_controls(gain, bass, mid, treble, pres, bright_on);

    const float bloom_atk = alpha_from_sec(0.030f, kDspOsFs);
    const float bloom_rel = alpha_from_sec(0.190f, kDspOsFs);
    const float tight_atk = alpha_from_sec(0.002f, kDspOsFs);
    const float tight_rel = alpha_from_sec(0.065f, kDspOsFs);
    const float pwr_atk = alpha_from_sec(0.012f, kDspOsFs);
    const float pwr_rel = alpha_from_sec(0.120f, kDspOsFs);

    // Level-matched progressive cascade: clean/blues low -> singing liquid lead mid -> surgical high-gain shred
    const float g1_lin = 0.75f + gain * 1.45f;
    const float g2 = 0.90f + gain * 4.90f;
    const float g3 = 0.90f + gain * 3.55f;
    const float blend_45 = std::clamp((gain - 0.36f) / 0.64f, 0.0f, 1.0f);
    const float g4 = 1.55f + blend_45 * 4.30f;
    const float g5 = 1.30f + blend_45 * 2.80f;

    const float master_drive = amp_master_drive(master, 0.62f, 1.46f);
    const float blend_trim = 1.00f - 0.22f * blend_45;
    const float out_level = amp_block_volume_taper(level, 0.72f);

    for (int i = 0; i < n; ++i)
    {
      float x = std::isfinite(in[i]) ? std::clamp(in[i], -8.0f, 8.0f) : 0.0f;
      float x_os[2];
      os2x.upsample(x, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float pre = bright_boost.tick(pre_focus.tick(pre_dip.tick(pre_hpf.tick(x_os[s]))));
        float s1 = std::tanh(pre * g1_lin * 0.45f) / 0.45f;

        float s2 = triode_stage(s1, g2, 0.12f, v2_env, bloom_atk, bloom_rel);
        float s3 = triode_stage(s2, g3, 0.16f, v3_env, bloom_atk, bloom_rel);

        // Open interstage (115 Hz tight HPF, zero high-cut wall)
        float pre_hi = inter_hpf.tick(s3);

        // Stages 4 & 5: Surgical fine-grain Soldano/Triple-Crown precision clipping
        float s4 = triode_stage(pre_hi, g4, 0.05f, v4_env, tight_atk, tight_rel);
        float s5 = std::tanh(s4 * g5 * 0.90f) / 0.90f;
        float cf = triode_stage(s5, 0.95f, 0.04f, cf_env, tight_atk, tight_rel);

        float pre_out = pre_hi * (1.0f - blend_45) + cf * blend_45;

        const float abs_p = std::fabs(pre_out * master_drive);
        pwr_sag_env = (abs_p > pwr_sag_env) ? (pwr_atk * abs_p + (1.0f - pwr_atk) * pwr_sag_env)
                                            : (pwr_rel * abs_p + (1.0f - pwr_rel) * pwr_sag_env);
        const float rail = 1.0f / (1.0f + 0.12f * std::min(1.5f, pwr_sag_env));
        float pwr = rail * asym_tanh((pre_out * master_drive) / rail, 0.02f / rail);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = post_soldano_edge.tick(post_clean_dip.tick(nfb_punch.tick(os_out)));
      y = tone_stack.tick(y) * blend_trim * out_level;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.35f, 1.35f);
    }
  }
};

// 1992 Peavey 5150 "Block Letter" Head (Dual Independent Channels: ID 61 = Rhythm, ID 62 = LEAD)
// Calibrated for open bandwidth (~15 kHz) with 2x oversampled multi-stage triode cascade,
// 1992 V2B 39k cold-clipper asymmetry on LEAD, NFB Resonance (~72 Hz) & 4x6L6GC Post power saturation.
struct Pv5150DSP
{
  using Filter2 = ::Filter2;

  Oversampler2x os2x;
  Filter2 pre_hpf;
  Filter2 pre_mid_push;
  Filter2 bright_shelf;
  Filter2 inter_hpf;
  Filter2 inter_fizz;
  Filter2 nfb_resonance;
  GuitarToneStack4 tone_stack;

  float v1_env = 0.0f;
  float v2_env = 0.0f;
  float v3_env = 0.0f;
  float v4_env = 0.0f;
  float pwr_sag_env = 0.0f;

  bool initialized = false;
  bool last_is_lead = false;
  float last_gain_eq = -999.0f;
  float last_resonance = -999.0f;
  int last_bright = -1;

  void reset(double /*sr*/ = 48000.0)
  {
    os2x.reset();
    pre_hpf.reset();
    pre_mid_push.reset();
    bright_shelf.reset();
    inter_hpf.reset();
    inter_fizz.reset();
    nfb_resonance.reset();
    tone_stack.reset();
    v1_env = v2_env = v3_env = v4_env = pwr_sag_env = 0.0f;
    last_gain_eq = last_resonance = -999.0f;
    last_bright = -1;
    initialized = false;
  }

  void update_controls(bool is_lead, float pre_gain, float resonance,
                       float bass, float mid, float treble, float pres, bool bright)
  {
    if (!initialized || is_lead != last_is_lead)
    {
      reset();
      last_is_lead = is_lead;
      inter_hpf.set_hpf(kDspOsFs, is_lead ? 112.0f : 72.0f, 0.707f);
      // Gentle anti-fizz control above 5.2 kHz inside lead loop while keeping >14 kHz openness
      inter_fizz.set_high_shelf(kDspOsFs, 5400.0f, is_lead ? -3.2f : -1.2f, 0.75f);
      pre_mid_push.set_peaking(kDspOsFs, is_lead ? 1050.0f : 820.0f, 0.90f, is_lead ? 2.4f : 1.0f);
      initialized = true;
    }

    const int br = (!is_lead && bright) ? 1 : 0;
    if (br != last_bright || std::fabs(pre_gain - last_gain_eq) > 0.008f)
    {
      // Bright switch mimics 4700 pF treble-bleed cap across Rhythm Pre Gain pot (diminishes at max gain)
      const float br_db = br ? (4.4f * (1.0f - 0.65f * pre_gain)) : 0.0f;
      bright_shelf.set_high_shelf(kDspOsFs, 2350.0f, br_db, 0.80f);

      const float hpf_fc = is_lead ? (82.0f + pre_gain * 32.0f) : (56.0f + pre_gain * 24.0f);
      pre_hpf.set_hpf(kDspOsFs, hpf_fc, 0.707f);
      last_gain_eq = pre_gain;
      last_bright = br;
    }

    if (std::fabs(resonance - last_resonance) > 0.008f)
    {
      // Peavey 5150 NFB Resonance: sweeps 4x12 low-end cabinet thump @ ~72 Hz
      // 0% = tight/flat (0 dB), 50% default = balanced +4.2/+4.6 dB thump, 100% = massive +8.8/+9.4 dB
      const float max_res_db = is_lead ? 9.4f : 8.6f;
      const float res_db = resonance * max_res_db;
      nfb_resonance.set_peaking(kDspBaseFs, 72.0f, is_lead ? 1.12f : 1.05f, res_db);
      last_resonance = resonance;
    }

    ToneStackSpec spec{};
    spec.bass_fc = is_lead ? 112.0f : 116.0f;
    spec.bass_span_db = 24.0f;
    spec.mid_fc = is_lead ? 650.0f : 680.0f;
    spec.mid_q = is_lead ? 0.76f : 0.72f;
    spec.mid_span_db = 24.0f;
    spec.treble_fc = is_lead ? 1820.0f : 1860.0f;
    spec.treble_span_db = 24.0f;
    spec.pres_fc = is_lead ? 4150.0f : 3950.0f;
    spec.pres_span_db = 20.0f;
    tone_stack.update(spec, bass, mid, treble, pres);
  }

  void process(const float* in, float* out, int n, bool is_lead,
               float pre_gain_raw, float post_raw, float resonance_raw,
               float bass_raw, float mid_raw, float treble_raw, float pres_raw,
               bool bright_on)
  {
    const float pre_gain  = std::clamp(pre_gain_raw, 0.0f, 1.0f);
    const float post      = std::clamp(post_raw, 0.0f, 1.0f);
    const float resonance = std::clamp(resonance_raw, 0.0f, 1.0f);
    const float bass      = std::clamp(bass_raw, 0.0f, 1.0f);
    const float mid       = std::clamp(mid_raw, 0.0f, 1.0f);
    const float treble    = std::clamp(treble_raw, 0.0f, 1.0f);
    const float pres      = std::clamp(pres_raw, 0.0f, 1.0f);

    if (post <= 0.0f)
    {
      std::memset(out, 0, static_cast<size_t>(n) * sizeof(float));
      return;
    }

    update_controls(is_lead, pre_gain, resonance, bass, mid, treble, pres, bright_on);

    const float fast_atk  = alpha_from_sec(0.0025f, kDspOsFs);
    const float fast_rel  = alpha_from_sec(0.075f, kDspOsFs);
    const float bloom_atk = alpha_from_sec(0.018f, kDspOsFs);
    const float bloom_rel = alpha_from_sec(0.145f, kDspOsFs);
    const float pwr_atk   = alpha_from_sec(0.014f, kDspOsFs);
    const float pwr_rel   = alpha_from_sec(0.130f, kDspOsFs);

    // Preamp drive staging
    const float g1 = is_lead ? (1.15f + pre_gain * 3.10f) : (0.62f + pre_gain * 2.25f);
    const float g2 = is_lead ? (1.40f + pre_gain * 4.40f) : (0.78f + pre_gain * 2.65f);
    const float g3 = is_lead ? (1.55f + pre_gain * 3.85f) : (0.85f + pre_gain * 2.15f);
    const float g4 = is_lead ? (1.25f + pre_gain * 2.40f) : 1.0f;

    // Post knob controls both clean output level (0%..70% headroom) and 4x6L6GC power saturation (70%..100%)
    const float post_pwr_push = std::clamp((post - 0.70f) / 0.30f, 0.0f, 1.0f);
    const float master_drive = amp_master_drive(post, 0.74f, 1.92f) + 0.25f * post_pwr_push;
    const float out_trim = amp_block_volume_taper(post, is_lead ? 0.56f : 0.64f) * (1.0f / (1.0f + 0.35f * pre_gain));

    for (int i = 0; i < n; ++i)
    {
      float x = std::isfinite(in[i]) ? std::clamp(in[i], -8.0f, 8.0f) : 0.0f;
      float x_os[2];
      os2x.upsample(x, x_os);

      float os_out = 0.0f;
      for (int s = 0; s < 2; ++s)
      {
        float pre = bright_shelf.tick(pre_mid_push.tick(pre_hpf.tick(x_os[s])));
        float stage_out = 0.0f;

        if (!is_lead)
        {
          // Rhythm Channel: V1A -> V1B -> V6A (open American clean to muscular crunch)
          float s1 = triode_stage(pre, g1, 0.06f, v1_env, bloom_atk, bloom_rel);
          float tight = inter_hpf.tick(s1);
          float s2 = triode_stage(tight, g2, 0.14f, v2_env, fast_atk, fast_rel);
          float s3 = triode_stage(s2, g3, 0.18f, v3_env, fast_atk, fast_rel);
          stage_out = inter_fizz.tick(s3);
        }
        else
        {
          // Lead Channel: 5-stage cascade with iconic V2B 39k cold-clipper asymmetry
          float s1 = triode_stage(pre, g1, 0.08f, v1_env, fast_atk, fast_rel);
          float tight = inter_hpf.tick(s1);
          float s2 = triode_stage(tight, g2, 0.12f, v2_env, fast_atk, fast_rel);
          // V2B 39k cold clipper: strong asymmetric bias shifts positive half cleanly
          float s3 = triode_stage(s2, g3, 0.42f, v3_env, fast_atk, fast_rel);
          float s4 = triode_stage(inter_fizz.tick(s3), g4, 0.16f, v4_env, bloom_atk, bloom_rel);
          stage_out = std::tanh(s4 * 0.92f) / 0.92f;
        }

        // 4x6L6GC Push-Pull Power Stage with subtle bias/crossover character + rail sag
        const float drv_in = stage_out * master_drive;
        const float abs_p = std::fabs(drv_in);
        pwr_sag_env = (abs_p > pwr_sag_env) ? (pwr_atk * abs_p + (1.0f - pwr_atk) * pwr_sag_env)
                                            : (pwr_rel * abs_p + (1.0f - pwr_rel) * pwr_sag_env);
        const float rail = 1.0f / (1.0f + 0.11f * std::min(1.5f, pwr_sag_env));
        float pwr = rail * asym_tanh(drv_in / rail, 0.045f / rail);
        os_out = os2x.downsample_tick(pwr);
      }

      float y = nfb_resonance.tick(os_out);
      y = tone_stack.tick(y) * out_trim;
      if (!std::isfinite(y))
        y = 0.0f;
      out[i] = std::clamp(y, -1.35f, 1.35f);
    }
  }
};

int knob_value_to_index(float raw, int file_count, bool dynamic_steps = false)
{
  if (file_count <= 0)
    return -2;
  const float lo = knob_min();
  const float hi = knob_max();
  float t = (hi > lo) ? (raw - lo) / (hi - lo) : 0.0f;
  t = std::clamp(t, 0.0f, 1.0f);
  if (dynamic_steps)
  {
    int step = static_cast<int>(t * static_cast<float>(file_count - 1) + 0.5f);
    return std::clamp(step, 0, file_count - 1);
  }
  int step = static_cast<int>(t * static_cast<float>(kKnobSteps - 1) + 0.5f);
  step = std::clamp(step, 0, kKnobSteps - 1);
  return (step < file_count) ? step : -1;
}

// Max channels we ever need to handle (Gonkulator ABI reports 1 or 2).
constexpr int kMaxChannels = 2;

struct ModelState
{
  // One independent nam::DSP instance per channel -- NAM models carry
  // internal state (WaveNet hidden state, history ring buffers), so feeding
  // L then R through a single shared instance would corrupt both channels'
  // continuity. Dual-mono: same model config, independent per-channel state.
  std::unique_ptr<nam::DSP> dsp[kMaxChannels];
  std::atomic<bool> loading{false};
  std::atomic<bool> ready{false};

  std::atomic<bool> switching{false};
  std::atomic<int> active_index{-1};
  // NaN sentinel guarantees the very first process() call always triggers an
  // initial load attempt, regardless of whatever raw value the knob starts
  // at.
  std::atomic<float> last_seen_knob_raw{std::numeric_limits<float>::quiet_NaN()};
  // Debounce for switch_model_in_background -- see its own comment.
  std::atomic<int64_t> last_switch_attempt_ms{0};
  // Knob-settle tracking (see nam_process_gonk's own comment on the
  // "3 buzzes in a row" cascade this fixes) -- audio-thread-only, no atomic
  // needed beyond what's already required for cross-thread visibility of
  // active_index/pending_index elsewhere.
  int candidate_index = -1;
  int64_t candidate_since_ms = 0;

  // Duck-and-switch model transition (see nam_process_gonk's own comment).
  // The background load thread builds the new model into pending_dsp/
  // pending_index and only sets pending_ready -- it never touches dsp[]
  // directly. Only the AUDIO THREAD ever writes dsp[]/active_index/
  // fade_progress, which also fixes a latent thread-safety bug: previously
  // the background thread swapped s.dsp[ch] directly while the audio thread
  // could be mid-call on the very same object.
  std::unique_ptr<nam::DSP> pending_dsp[kMaxChannels];
  int pending_index = -1; // set before pending_ready; read only after
  std::atomic<bool> pending_ready{false};
  // 0=steady, 1=fading out the old model, 2=fading in the new one. Atomic
  // because switch_model_in_background (background thread) reads it to
  // avoid starting a new switch while a transition is still in flight;
  // fade_progress itself is audio-thread-only, no atomic needed.
  std::atomic<int> fade_state{0};
  int32_t fade_progress = 0;

  // Topology epoch (see g_topology_epoch) this instance last completed a
  // load/reload under. Compared against the live g_topology_epoch in
  // nam_process_gonk's steady-state path; a mismatch means another
  // instance has joined the board since, shrinking everyone's shared
  // per-block budget, so this instance is forced through the same duck-
  // and-reload cycle a knob-driven switch uses to recalibrate its quality
  // tier. Only ever written by the audio thread (at install time), so
  // relaxed load/store would suffice, but kept acquire/release to match
  // the rest of this struct's cross-thread-visible fields.
  std::atomic<int> calibrated_epoch{-1};
  // Set (audio-thread-only, no atomic needed -- same thread that reads it)
  // when the CURRENT fade-out/reload cycle was armed by a topology-epoch
  // staleness check rather than a genuine knob-driven model change. Bypasses
  // the "knob settled back on the same model, no background work needed"
  // shortcut so the cycle actually reaches switch_model_in_background and
  // recalculates the quality tier. Cleared only once the reload is actually
  // installed (not merely attempted), so the existing fade_state==3 retry
  // safety-net still re-requests a forced reload if the first attempt was
  // lost (debounced / lost the `switching` CAS).
  bool force_recalibrate = false;

  // Active 3-band Tone Stack + Presence + Tight Boost filters
  Biquad tone_bass[kMaxChannels];
  Biquad tone_mid[kMaxChannels];
  Biquad tone_treble[kMaxChannels];
  Biquad tone_presence[kMaxChannels];
  Biquad tight_boost[kMaxChannels];

  // Mesa Triple Crown (TC-100) & Mesa Lone Star additive per-channel DSP state
  MesaTripleCrownDSP mesa_tc[kMaxChannels];
  MesaLoneStarDSP mesa_ls[kMaxChannels];
  DumbleSSSDSP dumble_sss[kMaxChannels];
  DumbleODSDSP dumble_ods[kMaxChannels];
  ElmwoodBonnevilleDSP bonneville[kMaxChannels];
  IreneCleanDSP irene[kMaxChannels];
  JonOverdriveDSP jon[kMaxChannels];
  Pv5150DSP pv5150[kMaxChannels];

  // Which engine object (this_) this slot belongs to -- see state_for().
  std::atomic<void*> owner_this{nullptr};
};

// Anxiety OD's process() is hijacked on the SHARED CLASS VTABLE -- every
// instance of the pedal anywhere on the board calls into the same
// nam_process_gonk. A single global ModelState (the original design, when
// only one instance was ever tested) meant a second Anxiety OD instance
// would share the SAME loaded nam::DSP objects -- feeding two logically
// independent audio streams into one WaveNet's internal hidden state/history
// buffers as if they were one continuous stream. Confirmed on real hardware:
// adding a second instance produced "crazy noise", exactly the symptom of
// corrupted streaming-model state. Fixed by keying state per engine-object
// pointer instead of a single singleton.
//
// Lock-free fixed-size table (not a mutex-protected map): claimed by the
// first process() call from a never-before-seen this_, kept PERMANENTLY for
// the rest of the boot -- slots are never released, even after the owning
// pedal is deleted. This means the real consumption isn't "how many Anxiety
// OD instances exist at once" but "how many DISTINCT instances have EVER
// existed this boot", which climbs every time a pedal is added+deleted+
// re-added (e.g. during exploratory testing/undo-redo), not just from
// genuinely concurrent instances. Confirmed on real hardware: a single test
// session produced 11 distinct this_ pointers -- kMaxInstances=4 (sized for
// "a few pedals on a board at once") silently collapsed instances 5+ onto
// one shared fallback slot, reintroducing the exact cross-instance audio
// corruption the per-instance design was built to prevent. Sized generously
// now for a long exploratory session, not just a static board -- each
// unclaimed slot costs a few bytes of atomics, negligible.
constexpr int kMaxInstances = 64;
ModelState g_instances[kMaxInstances];

ModelState& state_for(void* this_)
{
  for (auto& inst : g_instances)
  {
    if (inst.owner_this.load(std::memory_order_acquire) == this_)
      return inst;
  }
  for (auto& inst : g_instances)
  {
    void* expected = nullptr;
    if (inst.owner_this.compare_exchange_strong(expected, this_, std::memory_order_acq_rel))
    {
      // A brand-new instance just joined the shared per-block CPU budget --
      // see g_topology_epoch's own comment. Lock-free increment only; the
      // per-model cache itself is left alone here (no mutex from the audio
      // thread) and instead goes stale lazily, tagged by epoch.
      g_topology_epoch.fetch_add(1, std::memory_order_release);
      return inst;
    }
  }
  return g_instances[kMaxInstances - 1]; // all slots claimed -- degrade, don't crash
}

// Superseded nam_process()/nam_process_gonk() hooks' state.
ModelState& state()
{
  static ModelState s;
  return s;
}

// Current, additive nam_process_naml() hook's state -- deliberately a
// SEPARATE instance from state() above. The additive engine object and a
// hijacked Gonkulator instance are independent objects (possibly both
// present at once during the transition off the hijack design), so they must
// never share a "currently loaded model"/"currently switching" state.
ModelState& state_naml()
{
  static ModelState s;
  return s;
}

void load_fixed_path_in_background()
{
  auto& s = state();
  bool expected = false;
  if (!s.loading.compare_exchange_strong(expected, true))
    return; // already loading (or loaded) -- don't spawn twice

  spawn_detached([&s]() {
    try
    {
      const auto path = std::filesystem::path(model_path());
      std::unique_ptr<nam::DSP> loaded[kMaxChannels];
      for (auto& d : loaded)
      {
        d = nam::get_dsp(path);
        // buffer size here is advisory (prewarm sizing); real calls pass the
        // actual per-block num_samples to process() regardless.
        d->Reset(kSampleRate, 128);
      }
      for (int i = 0; i < kMaxChannels; ++i)
        s.dsp[i] = std::move(loaded[i]);
      s.ready.store(true, std::memory_order_release);
    }
    catch (...)
    {
      // Leave ready=false: process() keeps passing audio through dry.
      // (No model file yet, bad JSON, unsupported architecture, etc.)
    }
  });
}

// Minimum time between switch attempts, regardless of how often process()
// sees a changed raw value. Defensive: if a knob-value field ever turns out
// to be some other, continuously-changing internal parameter (not a stable
// UI knob), every process() call would otherwise see raw != prev and fire a
// fresh switch attempt on every single audio block. A human turning a
// physical knob does not need more than a few switch attempts per second;
// anything faster than this is either noise or a wrong-offset assumption,
// not real input.
constexpr int64_t kMinSwitchIntervalMs = 250;

// How long the knob must stay resolved to the SAME zone before a switch is
// even attempted for it -- see nam_process_gonk's own comment on the
// "several buzzes in a row on one knob turn" cascade this fixes. A real
// turning gesture sweeps through every zone between start and end; without
// this, each zone briefly passed through could start (and sometimes finish
// and get installed, each with its own full duck-and-switch fade) before the
// knob moved on. 150ms is comfortably longer than the brief hand-deceleration
// pauses within a single continuous turn, but short enough not to feel
// laggy once the knob actually stops.
constexpr int64_t kKnobSettleMs = 150;

int64_t now_ms()
{
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// A2 models expose a discrete set of quality tiers (SlimmableModel::
// SetSlimmableSize, 0.0=minimum/"Lite" .. 1.0=maximum/"Full" -- see
// https://www.tone3000.com/guides/nam-a2-the-complete-guide). Forcing 0.0
// always (the first working fix for the real-hardware glitching) is
// needlessly conservative for models with headroom to spare. This benchmarks
// every tier this specific model actually offers (GetSlimmableSizeBreakpoints
// tells us how many exist -- not all models have the same count) by running
// real inference on silent synthetic blocks at the real hardware's observed
// block size, from highest quality down, and keeps the highest tier whose
// measured time stays safely under the real-time budget. Runs entirely on
// the background load thread (SetSlimmableSize is documented "thread-safe,
// not real-time-safe"), never the audio thread, so this adds zero real-time
// risk -- worst case it takes a bit longer to finish loading.
constexpr int kCalibrationBlockSize = 48; // matches observed real hardware numFrames
constexpr int kCalibrationIters = 8;
// Require measured time under 50% of budget, not 100% (originally 70%,
// tightened after real-hardware testing -- see below). Real playing
// conditions have more jitter/contention than a clean synthetic benchmark,
// and the whole point of this feature is avoiding glitches, not chasing the
// exact edge of what's possible.
constexpr double kCalibrationSafetyMargin = 0.5;

// Every Anxiety OD instance anywhere on the board shares the SAME per-block
// real-time deadline (they're all processed within the same audio callback)
// -- but each instance's calibration used to assume it was the ONLY user of
// that budget. Confirmed on real hardware: instance A calibrated fine alone
// (691us against a 1000us budget, passed its own 70%-margin check), but once
// instance B was also active, A's real measured cost was 1503us -- both
// instances' individually-"safe" costs simply added up past the shared
// deadline. Dividing the budget by how many instances are currently claimed
// (see state_for()) doesn't retroactively fix an already-loaded instance's
// choice, but ensures every NEW calibration accounts for the others that
// already exist.
int count_claimed_instances()
{
  int n = 0;
  for (auto& inst : g_instances)
    if (inst.owner_this.load(std::memory_order_acquire) != nullptr)
      ++n;
  return n;
}

double calibrate_slimmable_quality(nam::DSP& dsp, nam::SlimmableModel& slimmable)
{
  const int concurrent_instances = std::max(1, count_claimed_instances());
  const double budget_us = static_cast<double>(kCalibrationBlockSize) / kSampleRate * 1e6 / concurrent_instances;
  const auto breakpoints = slimmable.GetSlimmableSizeBreakpoints();
  const size_t num_tiers = breakpoints.size() + 1;

  // A silent buffer risks underestimating real cost if the model (or Eigen's
  // own codepaths) has any signal-dependent fast path for near-zero input --
  // a low-amplitude sine is closer to a real, continuously-varying guitar
  // signal and won't trigger such a shortcut.
  std::vector<float> test_signal(static_cast<size_t>(kCalibrationBlockSize));
  for (int i = 0; i < kCalibrationBlockSize; ++i)
    test_signal[static_cast<size_t>(i)] =
      0.3f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 220.0 * i / kSampleRate));
  std::vector<float> scratch(static_cast<size_t>(kCalibrationBlockSize), 0.0f);
  float* in_arr[1] = {test_signal.data()};
  float* out_arr[1] = {scratch.data()};

  for (size_t tier = num_tiers; tier-- > 0;) // highest quality first
  {
    const double ratio = (static_cast<double>(tier) + 0.5) / static_cast<double>(num_tiers);
    slimmable.SetSlimmableSize(ratio);
    dsp.Reset(kSampleRate, kCalibrationBlockSize); // re-warm at this tier before timing

    int64_t total_us = 0;
    for (int iter = 0; iter < kCalibrationIters; ++iter)
    {
      const auto t0 = std::chrono::steady_clock::now();
      dsp.process(in_arr, out_arr, kCalibrationBlockSize);
      const auto t1 = std::chrono::steady_clock::now();
      total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      usleep(500);
    }
    const double avg_us = static_cast<double>(total_us) / static_cast<double>(kCalibrationIters);

    if (avg_us < budget_us * kCalibrationSafetyMargin)
      return ratio;
  }
  return 0.0; // nothing fit safely -- fall back to minimum/Lite
}

// Kicks off (at most one concurrent, rate-limited) background switch:
// re-constructs a nam::DSP from the already-in-memory cached JSON (see
// g_cached_models) and atomically swaps it in. Touches no filesystem at
// all -- only runs on a detached worker thread regardless.
//
// `force_reload_same_model`: bypasses the "knob landed back on the model
// already active -- nothing to do" early-return below, for a topology-
// triggered retroactive recalibration (see ModelState::force_recalibrate)
// that deliberately wants to reconstruct the SAME model, so its quality
// tier gets recomputed against the (now smaller) per-instance budget --
// see calibrate_slimmable_quality/g_topology_epoch.
void switch_model_in_background(ModelState& s, float raw, bool force_reload_same_model = false, bool dynamic_steps = false)
{
  const int64_t now = now_ms();
  const int64_t last = s.last_switch_attempt_ms.load(std::memory_order_relaxed);
  if (now - last < kMinSwitchIntervalMs)
    return; // debounced -- too soon since the last attempt
  s.last_switch_attempt_ms.store(now, std::memory_order_relaxed);

  if (!g_models_ready.load(std::memory_order_acquire))
  {
    preload_models_in_background(); // no-op if already started -- see its own comment
    return; // not finished yet -- next raw-change (or retry, see
            // nam_process_gonk) will try again
  }

  // Don't start a new background load while a previous one's result is
  // still being duck-and-switched in by the audio thread, or is sitting
  // unpicked-up in the pending slot -- see ModelState's own comment on why
  // dsp[]/pending_dsp[] are each owned by exactly one thread. fade_state==3
  // (already muted, waiting on an earlier request) is allowed through: the
  // pickup block's stale-pending-drop path calls back in here to replace a
  // load the knob has already moved past, while staying muted throughout.
  const int fade_state_now = s.fade_state.load(std::memory_order_relaxed);
  if ((fade_state_now != 0 && fade_state_now != 3) || s.pending_ready.load(std::memory_order_acquire))
    return;

  bool expected = false;
  if (!s.switching.compare_exchange_strong(expected, true))
    return; // a switch is already in flight; the next process() call that
            // still sees a changed value will retry once this one finishes

  spawn_detached([&s, raw, force_reload_same_model, dynamic_steps]() {
    // Everything in this thread body runs inside this try/catch, on purpose:
    // an exception escaping a detached thread's entry function calls
    // std::terminate() *without* unwinding (the Reset guard below would never
    // run), which not only kills the whole process but was observed to hang
    // rather than cleanly abort under QEMU-user ARM emulation.
    flush_denormals_to_zero(); // brand-new thread -- see its own comment
    lower_background_thread_priority(); // ditto -- see its own comment
    try
    {
      struct Reset
      {
        std::atomic<bool>& flag;
        ~Reset() { flag.store(false, std::memory_order_release); }
      } reset{s.switching};

      const int idx = knob_value_to_index(raw, static_cast<int>(g_cached_models.size()), dynamic_steps);
      if (idx == -2 || (idx == s.active_index.load(std::memory_order_acquire) && !force_reload_same_model))
        return; // no files found at all, or knob moved but landed back on the same state
                // (unless a forced recalibration deliberately wants this same model rebuilt)

      if (idx == -1)
      {
        // This step has no file assigned -- target is silence. Nothing to
        // build/prewarm/calibrate; just stage the silent target so the audio
        // thread ducks out the current model and swaps to nothing (see
        // nam_process_gonk's fade-complete swap and fade_state==0 branch).
        for (int i = 0; i < kMaxChannels; ++i)
          s.pending_dsp[i].reset();
        s.pending_index = idx;
        s.pending_ready.store(true, std::memory_order_release);
        return;
      }
      const nlohmann::json& config = g_cached_models[static_cast<size_t>(idx)].config;

      std::unique_ptr<nam::DSP> loaded[kMaxChannels];
      double chosen_ratio = 1.0;
      constexpr int32_t kPrewarmSamples = 16384;
      constexpr int32_t kPrewarmBlockSize = 128;
      for (int i = 0; i < kMaxChannels; ++i)
      {
        auto& d = loaded[i];
        d = nam::get_dsp(config);
        usleep(500);
        d->Reset(kSampleRate, 128); // establish sample rate/buffer size FIRST --
                                     // see the explicit manual prewarm below for
                                     // why this must happen before, not after,
                                     // any calibration/warm-up processing.
        // A2 models can run "Full" (desktop CPU) or "Lite" (pedalboard-class
        // hardware) at a large cost difference for the same capture -- see
        // https://www.tone3000.com/guides/nam-a2-the-complete-guide -- via
        // this runtime SlimmableModel interface (0.0=minimum/Lite,
        // 1.0=maximum/Full, and SlimmableWavenet defaults to 1.0/Full unless
        // told otherwise). Real-hardware profiling confirmed Full mode
        // running ~50% over this device's real-time per-block budget even
        // with -O3/-ffast-math -- a genuine compute ceiling this device's
        // ARM core can't close, not a bug. Non-slimmable models (plain
        // WaveNet/LSTM/etc, dynamic_cast fails) are unaffected -- this is a
        // no-op for them.
        if (auto* slimmable = dynamic_cast<nam::SlimmableModel*>(d.get()))
        {
          // Calibrate once (channel 0's instance -- same model/config, so
          // the same tier is safe for every channel) instead of forcing the
          // lowest tier always; apply the result to every channel afterward.
          // Cached per-model (see g_calibration_cache's own comment) --
          // real hardware showed glitches/occasional full-device reboots
          // when repeatedly switching, and redoing this full benchmark (2x
          // internal sub-model reconstruction per tier tested) on every
          // single switch was pure avoidable churn once a model has already
          // been calibrated once.
          if (i == 0)
          {
            std::lock_guard<std::mutex> lock(g_calibration_cache_mutex);
            auto& entry = g_calibration_cache[static_cast<size_t>(idx)];
            const int live_epoch = g_topology_epoch.load(std::memory_order_acquire);
            // Stale-by-epoch counts as a miss too, not just never-calibrated
            // -- see CalibrationEntry's own comment.
            if (entry.ratio < 0.0 || entry.epoch != live_epoch)
            {
              entry.ratio = calibrate_slimmable_quality(*d, *slimmable);
              entry.epoch = live_epoch;
            }
            else
              slimmable->SetSlimmableSize(entry.ratio);
            chosen_ratio = entry.ratio;
          }
          else
            slimmable->SetSlimmableSize(chosen_ratio);
        }

        // calibrate_slimmable_quality (above, channel-0/cache-miss path only)
        // calls dsp.Reset(kSampleRate, kCalibrationBlockSize) once per tier
        // while benchmarking, which -- same as any Reset() call -- resizes
        // mMaxBufferSize to that tier's benchmark block size (48), not back to
        // the 128 set at the top of this loop. Confirmed via host-side wav
        // testing (no device/emulation needed to hit this): the very first
        // time ANY SlimmableModel-architecture .nam file is ever selected
        // (first cache miss), the manual prewarm loop right below then calls
        // process() with kPrewarmBlockSize=128 against a model whose
        // mMaxBufferSize is still 48 from calibration's last Reset --
        // `assert(num_frames <= mMaxBufferSize)` in wavenet/model.cpp fires
        // immediately. Worse than a normal crash: this runs on a
        // pthread_create'd thread (see spawn_detached's own comment), so
        // assert()'s abort() is NOT a catchable C++ exception -- the
        // surrounding try/catch never sees it, and abort() takes down the
        // WHOLE PROCESS, not just this thread. Almost certainly a real
        // contributor to the "occasional full-device reboots when repeatedly
        // switching models" noted above (that comment blamed only CPU/timing
        // churn) -- and matches the "occasional" pattern exactly, since only
        // a first-time (cache-miss) selection of a given slimmable model hits
        // it; switching back to an already-calibrated one does not re-Reset
        // and never shrinks the buffer. Unconditional and harmless for
        // non-slimmable models too (mMaxBufferSize is already 128 for them;
        // this is a no-op resize to the same value).
        d->Reset(kSampleRate, kPrewarmBlockSize);

        // Explicit manual prewarm: process silence through the model to let
        // its internal WaveNet history/hidden state settle BEFORE any real
        // (or duck-and-switch fade-in) audio ever reaches it. DSP::Reset()
        // normally does this automatically via prewarm(), but
        // SlimmableWavenet explicitly overrides GetPrewarmSamples() to
        // return 0 (no automatic prewarm at all) -- confirmed in
        // NAM/wavenet/slimmable.h. Without this, the model's FIRST real
        // samples (right at the start of the fade-in) come from a "cold"
        // model with empty history, audible as a click/glitch on real
        // hardware. ~340ms of silence at 48kHz is comfortably more than any
        // realistic guitar-amp WaveNet's receptive field.
        {
          std::vector<float> silence(kPrewarmBlockSize, 0.0f);
          std::vector<float> scratch(kPrewarmBlockSize, 0.0f);
          float* in_arr[1] = {silence.data()};
          float* out_arr[1] = {scratch.data()};
          // Real-hardware symptom: audible cracks/artifacts during this
          // window even though nam_process_gonk now mutes the audio thread
          // (fade_state==3, plain silence, no dsp->process() calls at all)
          // for the ENTIRE duration of this loop -- so the audio thread
          // itself has nothing expensive to do, yet still isn't getting
          // scheduled reliably. On a single ARM core, a tight, syscall-free
          // loop only actually gets preempted at a scheduler tick boundary
          // (can be several ms on an embedded kernel's HZ), regardless of
          // this thread's niceness -- niceness affects how MUCH CPU share a
          // thread gets over time, not how promptly it yields the CPU
          // mid-burst. Real hardware's own audio callback period is ~1ms
          // (kCalibrationBlockSize=48 "matches observed real hardware
          // numFrames"), so any uninterrupted burst much longer than that
          // risks missing its wakeup -- an underrun/dropout, heard as a
          // crack. Yielding every 8 blocks (~21ms bursts) was too coarse;
          // yield after EVERY block instead (~2.7ms bursts, close to the
          // real audio period) so the gap between forced yields is never
          // much bigger than one real audio callback's worth of work.
          constexpr int32_t kPrewarmSlice = 48;
          std::vector<float> silence48(kPrewarmSlice, 0.0f);
          std::vector<float> scratch48(kPrewarmSlice, 0.0f);
          float* in_arr48[1] = {silence48.data()};
          float* out_arr48[1] = {scratch48.data()};
          for (int32_t done = 0; done < kPrewarmSamples; done += kPrewarmSlice)
          {
            d->process(in_arr48, out_arr48, kPrewarmSlice);
            usleep(500);
          }
        }
      }
      // Stage into pending_dsp, NOT dsp[] directly -- the audio thread
      // (nam_process_gonk) is the only thing that ever installs into dsp[],
      // once it's ready to duck-and-switch. See ModelState's own comment.
      for (int i = 0; i < kMaxChannels; ++i)
        s.pending_dsp[i] = std::move(loaded[i]);
      s.pending_index = idx;
      s.pending_ready.store(true, std::memory_order_release);
    }
    catch (...)
    {
      // Bad/corrupt cached config, unsupported architecture, etc. -- keep
      // whatever model was previously active (or stay in passthrough if
      // none has loaded yet). s.switching is still reset by the Reset guard
      // during this catch's unwind.
    }
  });
}

} // namespace

// SUPERSEDED reference implementation -- see file header. Not used by the
// shipped design, kept for the proven IRLoader-hook fallback path.
extern "C" void nam_process(void* this_)
{
  flush_denormals_to_zero(); // see its own comment -- Evil's real-time audio
                              // thread, not one we spawned ourselves
  auto* self = reinterpret_cast<uint8_t*>(this_);
  float* input = *reinterpret_cast<float**>(self + 0x68);
  float* output = *reinterpret_cast<float**>(self + 0x6c);
  const int32_t n = *reinterpret_cast<int32_t*>(self + 0x70);

  if (n <= 0 || !input || !output)
    return;

  auto& s = state();
  if (!s.ready.load(std::memory_order_acquire))
  {
    load_fixed_path_in_background();
    std::memcpy(output, input, static_cast<size_t>(n) * sizeof(float));
    return;
  }

  float* in_arr[1] = {input};
  float* out_arr[1] = {output};
  s.dsp[0]->process(in_arr, out_arr, n);
}

// Current design: replaces AnxietyOD engine's process(). See file header for
// the confirmed ABI. flags/ctx are intentionally left untouched -- the
// original's tail-flush/latency bookkeeping on *flags is not replicated in
// this v1 (an overdrive is not a reverb/delay with a real tail, so this is
// expected to be a safe simplification, but is an explicit known
// limitation, not a verified-safe one).
//
// Model selection: Drive's raw float value (this_+0x534) is read directly
// off the engine object -- confirmed via live-hardware wide-memory diffing
// (see patch_gonkulator.py's docstring and git history for the full
// derivation). No second trampoline/vtable-patch is needed just for knob
// tracking. Tone (0x548) and Level (0x520) feed input/output trim; bypass
// is this_+0x26b. See nam_process_gonk below for how each is used.

// Model-switch transition used to use one LINEAR ramp length for both
// halves (fade-out and fade-in). A linear ramp's gain is flat (slope 0)
// right up until the fade starts, then jumps to a constant nonzero slope --
// a discontinuity in the slope itself, which is an impulsive, broadband
// ("click"/"zz") artifact independent of whatever the model is doing.
// Shortening that old single length 4800->128 samples (~2.7ms) was flashed
// to real hardware and made the click WORSE, not better -- consistent with
// this theory (steeper ramp = bigger slope jump = louder click), not with
// "fade wasn't short enough yet". Do not reintroduce a single-length linear
// ramp; see raised_cosine_ramp() below, which has zero slope at both
// endpoints and removes the discontinuity regardless of length.
//
// Fade-out and fade-in are independently sized, both 100ms:
// - Fade-out: tried short (~5ms) once the ramp-shape fix above stopped it
//   clicking on its own, but real hardware still showed artifacts -- see
//   nam_process_gonk's own comment: the background load/calibrate/prewarm is
//   now deliberately NOT started until fade-out has fully finished (rather
//   than concurrently with it), so a slower, unhurried 100ms duck costs
//   nothing extra in when the new model becomes available, and avoids
//   whatever a too-fast duck of a still-live model was doing on its own.
// - Fade-in (100ms): the new model was
//   prewarmed on SILENCE (see switch_model_in_background's manual prewarm
//   block) before this fade-in ever starts, so it's not truly "cold" --
//   still, ramping the model's INPUT, not just its output (see the
//   fade_state==2 branch below), avoids handing it an instant full-amplitude
//   step, which is a plausible source of a small transient/overshoot even
//   from a silence-settled state. 100ms is a starting guess, not yet
//   hardware-confirmed at this specific value -- shorten via
//   NAM_FADE_IN_LEN_SAMPLES for host-side comparison, but only trust a real
//   flash+listen test before shipping a change here.
//
// Both overridable via NAM_FADE_OUT_LEN_SAMPLES / NAM_FADE_IN_LEN_SAMPLES --
// host-side wav testing only, to sweep these without a hardware/QEMU round
// trip each time. Do not lower the shipped defaults without re-confirming on
// real hardware, not just host wav sweeps -- see the linear-ramp history
// above for why host-side sweeps alone were previously misleading.
int32_t fade_out_len_samples()
{
  if (const char* env = std::getenv("NAM_FADE_OUT_LEN_SAMPLES"))
    return std::max(1, std::atoi(env));
  return 4800; // 100ms at 48kHz
}

int32_t fade_in_len_samples()
{
  if (const char* env = std::getenv("NAM_FADE_IN_LEN_SAMPLES"))
    return std::max(1, std::atoi(env));
  return 4800; // 100ms at 48kHz
}

// Equal-power-ish raised-cosine ramp: 0 at progress<=0, 1 at progress>=len,
// smooth (zero slope) at both ends. Used for both fade-out (as 1-this) and
// fade-in so neither transition has the linear ramp's slope discontinuity.
float raised_cosine_ramp(int32_t progress, int32_t len)
{
  const float x = std::clamp(static_cast<float>(progress) / static_cast<float>(len), 0.0f, 1.0f);
  return 0.5f * (1.0f - std::cos(static_cast<float>(M_PI) * x));
}

extern "C" void nam_process_gonk(void* this_, uint32_t /*param2*/, float** input, uint32_t numChannels,
                                  float** output, int32_t numFrames, uint32_t* /*flags*/, void* /*ctx*/)
{
  flush_denormals_to_zero(); // see its own comment -- Evil's real-time audio
                              // thread, not one we spawned ourselves
  if (numFrames <= 0 || !input || !output || numChannels == 0)
    return;
  if (numChannels > kMaxChannels)
    numChannels = kMaxChannels;

  auto& s = state_for(this_); // per-engine-instance state -- see its own comment
  const int32_t kFadeOutLenSamples = fade_out_len_samples();
  const int32_t kFadeInLenSamples = fade_in_len_samples();

  const float drive_raw = *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(this_) + 0x534); // model select
  const float tone_raw = *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(this_) + 0x548);   // input trim
  const float level_raw = *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(this_) + 0x520);  // output trim
  const uint8_t bypass_byte = *(reinterpret_cast<uint8_t*>(this_) + 0x26b);

  // Bypass is checked further down now, right before the first place it
  // actually needs to change behavior (the produced audio) -- NOT here.
  // Checking it here and returning early used to mean the knob-change
  // detection/switch-trigger/pending-pickup block below never ran at all
  // while the pedal sat bypassed, so a model never even started loading
  // until the user's first non-bypass engage -- the exact moment most
  // likely to be heard, since real audio is already flowing by then. Anxiety
  // OD commonly sits bypassed on a board from power-on until footswitched,
  // so this was effectively still a lazy-at-first-use load in disguise, not
  // eager-at-boot. Running this block regardless of bypass costs nothing
  // (no audio is produced from it either way) and means the load/duck-and-
  // switch/prewarm all happen in the background WHILE bypassed, finishing
  // (or getting much closer to finished) before the user ever un-bypasses.
  s.last_seen_knob_raw.store(drive_raw, std::memory_order_release);
  bool ready = s.ready.load(std::memory_order_acquire);
  if (!ready)
  {
    // Unconditional retry every call while not yet ready -- no settle
    // delay needed here, nothing is playing yet to protect from a
    // premature switch, so get the first model in as fast as possible.
    // switch_model_in_background's own kMinSwitchIntervalMs debounce (and
    // its `switching` in-flight guard) already makes calling this on
    // every block cheap and safe.
    //
    // Deliberately NOT gated on drive_raw having changed since the last
    // call (an earlier version was, "if changed-or-first-ever-call") --
    // confirmed on real hardware as a real "model never loads" bug: if
    // the knob sits perfectly still from power-on (the common case -- no
    // footswitch engaged yet, nobody touching Drive), the very first
    // process() call kicks off preload_models_in_background() and
    // returns (see switch_model_in_background's own !g_models_ready
    // branch) without yet having a model to install. With the old
    // change-gated retry, no later call would ever try again, since
    // drive_raw never differs from the value already stored as `prev` --
    // permanently stuck dry-passthrough. Retrying unconditionally instead
    // means the very next call after preload finishes picks it up.
    switch_model_in_background(s, drive_raw, s.force_recalibrate);
  }
  else
  {
    // Only start a background load once the knob has RESOLVED TO, AND
    // STAYED ON, the same zone for kKnobSettleMs -- see that constant's own
    // comment. Without this, a real turning gesture sweeping from one zone
    // to a distant one passes through every zone in between, and each one
    // it merely passed through could still start its own load; if that load
    // happened to finish before the knob moved on again, the drop-if-stale
    // check below has nothing to catch (the knob briefly WAS at that zone
    // when it was picked up) and it gets installed with a full audible
    // duck-and-switch fade -- heard on real hardware as several distinct
    // "buzz" transitions in a row from one single knob turn.
    const int target_idx = knob_value_to_index(drive_raw, static_cast<int>(g_cached_models.size()));
    if (target_idx != s.candidate_index)
    {
      s.candidate_index = target_idx;
      s.candidate_since_ms = now_ms();
    }
    else if (target_idx != -2 && target_idx != s.active_index.load(std::memory_order_acquire)
             && now_ms() - s.candidate_since_ms >= kKnobSettleMs)
    {
      // Start ducking the about-to-be-replaced model out RIGHT NOW --
      // deliberately NOT calling switch_model_in_background here yet. That
      // background construct/calibrate/prewarm work (still real, audible
      // artifacts even at a fast duck when it overlapped a still-live old
      // model, per real-hardware testing) is now only kicked off once this
      // fade-out has fully completed and the old model is truly silent --
      // see the fade_state==1 completion block below, which calls
      // switch_model_in_background itself once fade_progress reaches
      // kFadeOutLenSamples.
      if (s.fade_state.load(std::memory_order_relaxed) == 0)
      {
        s.fade_state.store(1, std::memory_order_relaxed);
        s.fade_progress = 0;
      }
    }

    // Retroactive quality recalibration: the shared per-instance CPU budget
    // only shrinks when a NEW instance joins the board (see
    // g_topology_epoch) -- nothing above notices that an already-playing,
    // already-steady instance's quality tier was chosen for a larger, now
    // stale budget. If this instance is steady (fade_state==0, nothing
    // already in flight -- including whatever the knob-settle check above
    // may have just armed this same call) and has a real model loaded,
    // force it through the exact same duck-out/reload/duck-in cycle a
    // knob-driven switch uses, just re-targeting its OWN current model --
    // switch_model_in_background's epoch-tagged cache check (see
    // CalibrationEntry) then recalculates its tier against the new,
    // smaller budget instead of reusing the stale cached one.
    if (s.fade_state.load(std::memory_order_relaxed) == 0 && s.active_index.load(std::memory_order_acquire) >= 0
        && s.calibrated_epoch.load(std::memory_order_acquire) != g_topology_epoch.load(std::memory_order_acquire))
    {
      s.force_recalibrate = true;
      s.fade_state.store(1, std::memory_order_relaxed);
      s.fade_progress = 0;
    }
  }

  // Safety net against a permanent-mute trap: fade_state==3 is only ever
  // entered alongside a single one-shot switch_model_in_background call (see
  // the fade_state==1 completion block below). If that ONE call fails to
  // ever deliver a result -- debounced by kMinSwitchIntervalMs, lost a CAS
  // race on `switching` to an unrelated overlapping attempt, or the
  // background thread's try/catch swallowed a load exception (corrupt
  // config, bad_alloc, unsupported architecture) -- nothing else in this
  // function ever retries it: the pickup block below only acts once
  // pending_ready is already true, and the knob-settle branch above only
  // arms a new transition from fade_state==0, not 3. Without this, the
  // instance stays muted forever, matching the real-world report of an
  // instance going silent and never recovering even across NAM-file changes
  // or new rigs. Mirrors the unconditional per-call retry already used for
  // the !ready bootstrap case above; switch_model_in_background's own
  // debounce/CAS guards make calling it every block here cheap and safe.
  if (s.fade_state.load(std::memory_order_relaxed) == 3 && !s.pending_ready.load(std::memory_order_acquire))
    switch_model_in_background(s, drive_raw, s.force_recalibrate);

  // Pick up a finished background load, if any. Only the audio thread ever
  // installs into dsp[] -- see ModelState's own comment. fade_state==3 means
  // the old model already faded all the way out and the background thread
  // was kicked off only then (see fade_state==1's completion above);
  // fade_state==0 covers only the first-ever load (nothing was playing to
  // duck, so no fade-out/mute step exists for it).
  const int fs_for_pickup = s.fade_state.load(std::memory_order_relaxed);
  if ((fs_for_pickup == 0 || fs_for_pickup == 3) && s.pending_ready.load(std::memory_order_acquire))
  {
    // The knob can move on while a load (esp. a first-time-this-boot
    // calibration, which is much slower than a cached one) is still in
    // flight -- discovered on real hardware as a fast knob sweep producing
    // several audible duck-and-switch cycles in a row, each landing on a
    // model the knob had already moved past by the time it was ready. Only
    // ever install the load that still matches where the knob actually is
    // right now; a stale one is silently dropped (no fade played for it)
    // and immediately replaced by a fresh request for the current target.
    const int current_idx = knob_value_to_index(drive_raw, static_cast<int>(g_cached_models.size()));
    if (ready && current_idx != -2 && current_idx != s.pending_index)
    {
      for (int i = 0; i < kMaxChannels; ++i)
        s.pending_dsp[i].reset();
      s.pending_ready.store(false, std::memory_order_release);
      switch_model_in_background(s, drive_raw, s.force_recalibrate);
      // If fs_for_pickup==3 we're already muted -- stay that way until the
      // fresh request above lands here again. If fs_for_pickup==0, nothing
      // audible is playing here either way (see comment above), so no fade
      // is needed just for dropping the stale load.
    }
    else if (!ready)
    {
      // First-ever load for this instance -- nothing currently playing to
      // duck, so install directly and skip the fade entirely.
      for (int i = 0; i < kMaxChannels; ++i)
        s.dsp[i] = std::move(s.pending_dsp[i]);
      s.active_index.store(s.pending_index, std::memory_order_release);
      s.pending_ready.store(false, std::memory_order_release);
      s.ready.store(true, std::memory_order_release);
      ready = true;
      s.calibrated_epoch.store(g_topology_epoch.load(std::memory_order_acquire), std::memory_order_release);
      s.force_recalibrate = false;
    }
    else // fs_for_pickup == 3, ready == true, not stale -- the only remaining case
    {
      // Old model already faded all the way out before this load was even
      // requested (see fade_state==1's completion above) -- swap in the new
      // model and fade it straight IN. No fade-OUT needed here.
      for (int i = 0; i < kMaxChannels; ++i)
        s.dsp[i] = std::move(s.pending_dsp[i]);
      s.active_index.store(s.pending_index, std::memory_order_release);
      s.pending_ready.store(false, std::memory_order_release);
      s.fade_progress = 0;
      s.fade_state.store(s.pending_index < 0 ? 0 : 2, std::memory_order_relaxed);
      s.calibrated_epoch.store(g_topology_epoch.load(std::memory_order_acquire), std::memory_order_release);
      s.force_recalibrate = false;
    }
  }

  // Bypass takes precedence over everything from here on, including
  // "not ready yet" -- dry passthrough unconditionally. Deliberately checked
  // here, AFTER the load/switch/pending-pickup block above, not before it --
  // see that block's own comment.
  if (bypass_byte != 0 || !ready)
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
      if (input[ch] && output[ch])
        std::memcpy(output[ch], input[ch], static_cast<size_t>(numFrames) * sizeof(float));
    return;
  }

  // Tone/Level are [0,1] with 0.5=center/unity, 0.0=silence -- trim_gain()
  // takes that range directly.
  const float in_gain = trim_gain(tone_raw);
  const float out_gain = trim_gain(level_raw);

  // Clamp the model's input feed to [-1,1] -- trim_gain() can boost input up
  // to +trim_max_db() (~4x at full boost), and a NAM model fed signal well
  // outside the level range it was captured/trained at responds with harsh,
  // sustained-sounding artifacts of its own (not a graceful overdrive), not
  // just a scaled-up version of its normal output. Different models have
  // different natural input sensitivity, so a boost that was fine for the
  // previous model can push a newly-switched-to model out of its trained
  // range -- reported on real hardware as distortion that persists after a
  // model switch, not just a transient at the switch itself.
  for (uint32_t ch = 0; ch < numChannels; ++ch)
    if (input[ch] && output[ch])
      for (int32_t i = 0; i < numFrames; ++i)
        output[ch][i] = std::clamp(input[ch][i] * in_gain, -1.0f, 1.0f);

  const int fade_state = s.fade_state.load(std::memory_order_relaxed);
  const bool silent_active = s.active_index.load(std::memory_order_relaxed) < 0;
  if (fade_state == 0)
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      if (silent_active)
      {
        // This knob step has no file assigned -- render silence, not the dry
        // signal the input-clamp loop above just wrote into output[ch].
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
        continue;
      }
      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);
    }
  }
  else if (fade_state == 1) // fading OUT the current (about-to-be-replaced) model
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      if (silent_active)
      {
        // Already silent (this step had no file assigned) -- nothing to duck
        // out of, and s.dsp[ch] is null. Just hold silence through the fade
        // window so the swap below fires on schedule.
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
        continue;
      }
      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);
      for (int32_t i = 0; i < numFrames; ++i)
      {
        const float g = 1.0f - raised_cosine_ramp(s.fade_progress + i, kFadeOutLenSamples);
        output[ch][i] *= g;
      }
    }
    s.fade_progress += numFrames;
    if (s.fade_progress >= kFadeOutLenSamples)
    {
      s.fade_progress = 0;
      // Old model is now fully silent -- re-check the knob's CURRENT
      // position (it can wander during the 100ms duck) before doing
      // anything else.
      const int current_idx = knob_value_to_index(drive_raw, static_cast<int>(g_cached_models.size()));
      if (current_idx == s.active_index.load(std::memory_order_acquire) && !s.force_recalibrate)
      {
        // Knob settled back on the model we just faded out -- it's still
        // loaded in dsp[] (never touched during the duck), so just fade it
        // back IN directly. No background work needed at all. (Skipped when
        // force_recalibrate is set -- a topology-triggered recalibration
        // deliberately wants this same model rebuilt at a recalculated
        // quality tier, not just faded back in unchanged.)
        s.fade_state.store(s.active_index.load(std::memory_order_relaxed) < 0 ? 0 : 2,
                            std::memory_order_relaxed);
      }
      else
      {
        // Only NOW -- with the old model fully silent, guaranteed no
        // overlap with anything the audio thread is doing -- kick off the
        // new model's construct/calibrate/prewarm. See fade_state==3 below
        // for the muted wait, and switch_model_in_background's own guard,
        // which allows starting from fade_state==3.
        s.fade_state.store(3, std::memory_order_relaxed);
        switch_model_in_background(s, drive_raw, s.force_recalibrate);
      }
    }
  }
  else if (fade_state == 3) // muted: old model already ducked out, new one still loading
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
      if (input[ch] && output[ch])
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
  }
  else // fade_state == 2: fading IN the newly-installed model
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      // Fade the model's INPUT in too, not just its output: the new model
      // was just prewarmed on silence (see switch_model_in_background), so
      // its internal state is cold. Feeding it a smooth ramp-up instead of
      // an instant full-amplitude step avoids shocking a nonlinear model
      // (WaveNet/LSTM) with a discontinuous input it never saw in training.
      // The output ramp is kept too (belt-and-suspenders against any
      // residual near-zero-input bias/noise floor in the model itself).
      for (int32_t i = 0; i < numFrames; ++i)
        output[ch][i] *= raised_cosine_ramp(s.fade_progress + i, kFadeInLenSamples);
      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);
      for (int32_t i = 0; i < numFrames; ++i)
      {
        const float g = raised_cosine_ramp(s.fade_progress + i, kFadeInLenSamples);
        output[ch][i] *= g;
      }
    }
    s.fade_progress += numFrames;
    if (s.fade_progress >= kFadeInLenSamples)
    {
      s.fade_progress = 0;
      s.fade_state.store(0, std::memory_order_relaxed); // transition complete
    }
  }

  // Same reasoning as the input clamp above: different .nam models have
  // different inherent output loudness, so an out_gain boost that didn't
  // clip the previous model can push a louder new model's output well past
  // +-1.0 -- with nothing downstream to catch it, that's raw float overflow
  // reaching the audio hardware, heard as sustained distortion for as long
  // as that model + trim setting are active, not just a switch-moment glitch.
  for (uint32_t ch = 0; ch < numChannels; ++ch)
    if (input[ch] && output[ch])
      for (int32_t i = 0; i < numFrames; ++i)
        output[ch][i] = std::clamp(output[ch][i] * out_gain, -1.0f, 1.0f);
}

// ---- Input/Output trim (current, additive design only) ----
//
// Two NEW custom parameter setters, engine vtable slots 14/16 (see
// patch_namloader.py) -- deliberately NOT the shared generic float setter
// (slot 15/0x1a3d04, which stays the model-select knob). Same confirmed
// setter ABI (this=engine object in r0, value in VFP s0), but these don't
// write into the engine object's own memory at all: its internal layout
// beyond the known +0x2ac model-select field isn't fully mapped, so writing
// anywhere else risks colliding with real internal state. Storing here
// instead is exactly as safe as the process() hook's own hook_slot
// indirection and needs no assumptions about engine object internals.
//
// Value convention: NOT yet confirmed against a real knob (no UI wiring
// exists yet -- see SESSION_NOTES.md Phase 3 status). Assumed bipolar,
// roughly [-1, +1], 0 = unity/center, matching the shared setter's own
// bipolar branch-on-sign shape (BREAKTHROUGH #4). Revisit once a real
// control is actually wired to these slots.
struct TrimState
{
  std::atomic<float> input_raw{0.0f};
  std::atomic<float> output_raw{0.0f};
};

TrimState& trim_state()
{
  static TrimState t;
  return t;
}

extern "C" void nam_set_input_trim(void* /*this_*/, float value)
{
  trim_state().input_raw.store(value, std::memory_order_release);
}

extern "C" void nam_set_output_trim(void* /*this_*/, float value)
{
  trim_state().output_raw.store(value, std::memory_order_release);
}

// Current, additive design: replaces OUR OWN new engine object's process().
// Same 8-arg ABI as nam_process_gonk (byte-for-byte cloned engine layout, see
// patch_namloader.py's BREAKTHROUGH #6 header) but a genuinely separate
// object -- Gonkulator's real class/vtable is never written to. flags/ctx
// left untouched, same known limitation as nam_process_gonk (see its own
// comment above; carries over unchanged since this is the same engine
// shape).
//
// Model selection: identical mechanism to nam_process_gonk (this_+0x2ac,
// engine vtable slot 15's shared setter, unchanged in our cloned vtable) --
// this is the shipped file-picker per the user's decision, not a stopgap.
extern "C" void nam_process_naml(void* this_, uint32_t /*param2*/, float** input,
                                  uint32_t numChannels, float** output, int32_t numFrames,
                                  uint32_t* /*flags*/, void* /*ctx*/)
{
  flush_denormals_to_zero(); // see its own comment -- Evil's real-time audio
                              // thread, not one we spawned ourselves
  if (numFrames <= 0 || !input || !output || numChannels == 0)
    return;
  if (numChannels > kMaxChannels)
    numChannels = kMaxChannels;

  auto& s = state_naml();

  if (this_)
  {
    const float raw = *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(this_) + 0x2ac);
    const float prev = s.last_seen_knob_raw.exchange(raw, std::memory_order_acq_rel);
    if (raw != prev || prev != prev)
      switch_model_in_background(s, raw);
  }

  auto& t = trim_state();
  const float in_gain = trim_gain(t.input_raw.load(std::memory_order_acquire));
  const float out_gain = trim_gain(t.output_raw.load(std::memory_order_acquire));

  if (!s.ready.load(std::memory_order_acquire))
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
      if (input[ch] && output[ch])
        std::memcpy(output[ch], input[ch], static_cast<size_t>(numFrames) * sizeof(float));
    return;
  }

  for (uint32_t ch = 0; ch < numChannels; ++ch)
  {
    if (!input[ch] || !output[ch])
      continue;
    // Apply input trim before NAM, output trim after -- our own design
    // choice (no further RE needed, per the approved plan).
    for (int32_t i = 0; i < numFrames; ++i)
      output[ch][i] = input[ch][i] * in_gain;
    float* in_arr[1] = {output[ch]};
    float* out_arr[1] = {output[ch]};
    s.dsp[ch]->process(in_arr, out_arr, numFrames);
    for (int32_t i = 0; i < numFrames; ++i)
      output[ch][i] *= out_gain;
  }
}

// Test/debug-only introspection, not used by the real trampoline hook path.
// Lets test harnesses synchronize on "the background switch has actually
// finished" instead of inferring it indirectly from output no longer being
// dry passthrough, which races with a switch still in flight (observed to
// cause a used-after-teardown crash in a test that exited main() while a
// switch thread was still running -- see SESSION_NOTES.md).
extern "C" int nam_debug_is_switching(void)
{
  return state().switching.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int nam_debug_active_index(void)
{
  return state().active_index.load(std::memory_order_acquire);
}

extern "C" int nam_debug_is_switching_naml(void)
{
  return state_naml().switching.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int nam_debug_active_index_naml(void)
{
  return state_naml().active_index.load(std::memory_order_acquire);
}

// Per-instance variants of the above, for nam_process_gonk's hijack path,
// which is keyed by engine-object pointer (state_for()), not the single
// state()/state_naml() singletons the four functions above read. Test/debug
// only, same as the rest of this section.
extern "C" int nam_debug_is_switching_for(void* this_)
{
  return state_for(this_).switching.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int nam_debug_active_index_for(void* this_)
{
  return state_for(this_).active_index.load(std::memory_order_acquire);
}

extern "C" int nam_debug_fade_state_for(void* this_)
{
  return state_for(this_).fade_state.load(std::memory_order_acquire);
}

extern "C" int nam_debug_pending_ready_for(void* this_)
{
  return state_for(this_).pending_ready.load(std::memory_order_acquire) ? 1 : 0;
}

// Original AmpCab process function in Evil (Gigboard 2.7)
typedef void (*amp_process_fn)(void*, uint32_t, float**, uint32_t, float**, int32_t, uint32_t*, void*);
static amp_process_fn orig_amp_process()
{
  static amp_process_fn fn = []() -> amp_process_fn {
    if (const char* env = std::getenv("NAM_ORIG_AMP_PROCESS_ADDR"))
    {
      uintptr_t addr = static_cast<uintptr_t>(std::strtoul(env, nullptr, 0));
      if (addr != 0)
        return reinterpret_cast<amp_process_fn>(addr);
    }
    return reinterpret_cast<amp_process_fn>(0x13a0a0);
  }();
  return fn;
}

extern "C" void nam_process_amp(void* this_, uint32_t param2, float** input, uint32_t numChannels,
                                float** output, int32_t numFrames, uint32_t* flags, void* ctx)
{
  flush_denormals_to_zero();
  if (numFrames <= 0 || !input || !output || numChannels == 0 || !this_)
    return;
  if (numChannels > kMaxChannels)
    numChannels = kMaxChannels;

  auto* self = reinterpret_cast<uint8_t*>(this_);

  // Inspect Amp Model Type (DSP ID) at offset 0x94ec (ChA) and 0x9548 (ChB).
  // "82 LEAD 800 Bright" resolves to DSP ID 27 (0x1b) on Gigboard 2.7 hardware
  // (or 37 / 0x25 depending on lookup path).
  // 54th Amp ("Mesa Triple Crown Ch2")  -> DSP ID 53 (0x35)
  // 55th Amp ("Mesa Lone Star")         -> DSP ID 54 (0x36)
  // 56th Amp ("Mesa Triple Crown Lead") -> DSP ID 55 (0x37)
  // 57th Amp ("Dumble Steel String")    -> DSP ID 56 (0x38)
  // 58th Amp ("Dumble ODS")             -> DSP ID 57 (0x39)
  // 59th Amp ("Elmwood Bonneville 50")  -> DSP ID 58 (0x3a)
  // 60th Amp ("IRENE")                  -> DSP ID 59 (0x3b)
  // 61st Amp ("JON")                    -> DSP ID 60 (0x3c)
  // 62nd Amp ("5150 Original 1992 Rhythm") -> DSP ID 61 (0x3d), UI 0x29
  // 63rd Amp ("5150 Original LEAD")        -> DSP ID 62 (0x3e), UI 0x2b
  const int32_t amp_type = *reinterpret_cast<int32_t*>(self + 0x94ec);
  const int32_t amp_type_b = *reinterpret_cast<int32_t*>(self + 0x9548);
  const int32_t custom_id =
      (amp_type >= 53 && amp_type <= 62) ? amp_type :
      (amp_type_b >= 53 && amp_type_b <= 62) ? amp_type_b : -1;
  const bool is_tc_id = (custom_id == 53 || custom_id == 55);
  const bool is_ls_id = (custom_id == 54);
  const bool is_sss_id = (custom_id == 56);
  const bool is_ods_id = (custom_id == 57);
  const bool is_bonn_id = (custom_id == 58);
  const bool is_irene_id = (custom_id == 59);
  const bool is_jon_id = (custom_id == 60);
  const bool is_5150_rhy_id = (custom_id == 61);
  const bool is_5150_lead_id = (custom_id == 62);
  const bool is_5150_id = (is_5150_rhy_id || is_5150_lead_id);

  // Mesa Triple Crown (CH2 / Lead), Lone Star, Dumble SSS, Dumble ODS, Elmwood Bonneville 50 & IRENE routing:
  // Preserves all 53 stock amps and existing 82 LEAD 800 Bright (27/37) NAM blocks.
  if (custom_id != -1 || (amp_type != 37 && amp_type != 27 && amp_type_b != 37 && amp_type_b != 27))
  {
    auto read_tc_param = [self](size_t off, float def_val = 0.5f) -> float {
      float v = *reinterpret_cast<float*>(self + off);
      if (std::isnan(v) || std::isinf(v)) return def_val;
      if (v > 1.0f && v <= 100.0f) return v / 100.0f;
      return std::clamp(v, 0.0f, 1.0f);
    };

    const float mid_freq_val = *reinterpret_cast<float*>(self + 0x950c);
    const float trem_depth_norm = read_tc_param(0x9538, 0.5f);
    const bool is_tc_mid = (std::fabs(mid_freq_val - 776.0f) < 1.5f) ||
                           (std::fabs(mid_freq_val - 0.20f) < 0.005f);
    int tc_channel = 0;
    if (is_tc_mid)
    {
      if (trem_depth_norm >= 0.105f && trem_depth_norm <= 0.115f) tc_channel = 1;
      else if (trem_depth_norm > 0.115f && trem_depth_norm <= 0.125f) tc_channel = 2;
      else if (trem_depth_norm > 0.125f && trem_depth_norm <= 0.135f) tc_channel = 3;
    }

    if (custom_id == 53) tc_channel = 2;       // Mesa Triple Crown Ch2
    else if (custom_id == 55) tc_channel = 3;  // Mesa Triple Crown Lead

    if (is_ls_id || is_sss_id || is_ods_id || is_bonn_id || is_irene_id || is_jon_id || is_5150_id || tc_channel > 0)
    {
      // HeadRush Evil offset 0x95a5: Bypass flag (0 == ON/Active, != 0 == Bypassed/Off)
      if (*(self + 0x95a5) != 0)
      {
        for (uint32_t ch = 0; ch < numChannels; ++ch)
        {
          if (output[ch] != input[ch])
            std::memcpy(output[ch], input[ch], static_cast<size_t>(numFrames) * sizeof(float));
        }
        return;
      }

      auto& s = state_for(this_);
      // Verified Amp1 offsets:
      //   UI 0x0f (IDs 53..60): 0x9504=GainB(Drive), 0x9500=GainA(Volume), 0x9508=Master, 0x9510=Presence
      //   UI 0x29 / 0x2b (5150 IDs 61..62): 0x9500=GainA(Pre Gain), 0x9504=GainB(Resonance),
      //                                     0x9508=Master(Post), 0x950c=Presence
      //   Shared: 0x9518=Bright, 0x9524=Treble(High), 0x9528=Mid, 0x952c=Bass(Low)
      const float gain_raw     = read_tc_param(0x9504, 0.5f);
      const float level_raw    = read_tc_param(0x9500, 0.5f);
      const float master_raw   = read_tc_param(0x9508, 0.5f);
      const float presence_raw = read_tc_param(is_5150_id ? 0x950c : 0x9510, 0.5f);
      const float treble_raw   = read_tc_param(0x9524, 0.5f);
      const float mid_raw      = read_tc_param(0x9528, 0.5f);
      const float bass_raw     = read_tc_param(0x952c, 0.5f);
      const bool bright_on     = (*(self + 0x9518) != 0);

      for (uint32_t ch = 0; ch < numChannels; ++ch)
      {
        if (is_5150_id)
        {
          // 5150 layout (0x29 / 0x2b): level_raw (0x9500) = Pre Gain,
          //                            master_raw (0x9508) = Post,
          //                            gain_raw (0x9504) = Resonance
          s.pv5150[ch].process(input[ch], output[ch], numFrames, is_5150_lead_id,
                               level_raw, master_raw, gain_raw,
                               bass_raw, mid_raw, treble_raw, presence_raw, bright_on);
        }
        else if (is_ls_id)
        {
          s.mesa_ls[ch].process(input[ch], output[ch], numFrames,
                                gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                                presence_raw, bright_on);
        }
        else if (is_sss_id)
        {
          s.dumble_sss[ch].process(input[ch], output[ch], numFrames,
                                   gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                                   presence_raw, bright_on);
        }
        else if (is_ods_id)
        {
          s.dumble_ods[ch].process(input[ch], output[ch], numFrames,
                                   gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                                   presence_raw, bright_on);
        }
        else if (is_bonn_id)
        {
          s.bonneville[ch].process(input[ch], output[ch], numFrames,
                                   gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                                   presence_raw, bright_on);
        }
        else if (is_irene_id)
        {
          s.irene[ch].process(input[ch], output[ch], numFrames,
                              gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                              presence_raw, bright_on);
        }
        else if (is_jon_id)
        {
          s.jon[ch].process(input[ch], output[ch], numFrames,
                            gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                            presence_raw, bright_on);
        }
        else
        {
          s.mesa_tc[ch].process(input[ch], output[ch], numFrames, tc_channel,
                                gain_raw, master_raw, level_raw, bass_raw, mid_raw, treble_raw,
                                presence_raw, bright_on);
        }
      }
      return;
    }
  }

  if (amp_type != 37 && amp_type != 27 && amp_type_b != 37 && amp_type_b != 27)
  {
    orig_amp_process()(this_, param2, input, numChannels, output, numFrames, flags, ctx);
    return;
  }

  auto& s = state_for(this_);
  const int32_t kFadeOutLenSamples = fade_out_len_samples();
  const int32_t kFadeInLenSamples = fade_in_len_samples();

  auto read_param = [self](size_t off, float def_val = 0.5f) -> float {
    float v = *reinterpret_cast<float*>(self + off);
    if (std::isnan(v) || std::isinf(v)) return def_val;
    if (v > 1.0f && v <= 100.0f) return v / 100.0f;
    return std::clamp(v, 0.0f, 1.0f);
  };

  // Hardware-verified AmpCab offsets (confirmed via live amp_debug telemetry,
  // evil.db preset schema, and disassembly of 0x145634 / 0x137e58):
  //   0x9500: Gain       (Page 1, Knob 1: 0.0..1.0, default 0.5)
  //   0x9510: Presence   (Page 1, Knob 2: 0.0..1.0, default 0.5)
  //   0x9508: Master     (Page 1, Knob 3: 0.0..1.0, default 0.5)
  //   0x9524: Bass       (Page 2, Knob 1: 0.0..1.0, default 0.5)
  //   0x9528: Mid        (Page 2, Knob 2: 0.0..1.0, default 0.5)
  //   0x952c: Treble     (Page 2, Knob 3: 0.0..1.0, default 0.5)
  //   0x9538: Depth      (Page 3, Knob 1: 0.0..1.0, default 0.5) -> NAM Model Selector
  //   0x9518: Bright     (Page 3, Switch: uint8_t, 0=Off, 1=On)  -> Pre-NAM Tight Cut HPF
  //   0x95a5: Bypass     (Block On/Off:   uint8_t, 0=On,  1=Bypassed/Off)

  const float gain_raw         = read_param(0x9500, 0.5f);
  const float presence_raw     = read_param(0x9510, 0.5f);
  const float master_raw       = read_param(0x9508, 0.5f);
  const float bass_raw         = read_param(0x9524, 0.5f);
  const float mid_raw          = read_param(0x9528, 0.5f);
  const float treble_raw       = read_param(0x952c, 0.5f);
  const float model_select_raw = read_param(0x9538, 0.5f);
  const bool bright_on         = (*(self + 0x9518) != 0);
  const uint8_t bypass_byte    = *(self + 0x95a5);

  const float mid_freq_hz = 750.0f; // Classic British Marshall mid sweet-spot for tone stack

  // Configure active tone-stack filters (at 0.5 / 50%, gain_db is 0.0 dB -> exact unity)
  for (uint32_t ch = 0; ch < numChannels; ++ch)
  {
    s.tone_bass[ch].set_low_shelf(static_cast<float>(kSampleRate), 100.0f, (bass_raw - 0.5f) * 24.0f);
    s.tone_mid[ch].set_peaking(static_cast<float>(kSampleRate), mid_freq_hz, (mid_raw - 0.5f) * 24.0f, 1.0f);
    s.tone_treble[ch].set_high_shelf(static_cast<float>(kSampleRate), 3500.0f, (treble_raw - 0.5f) * 24.0f);
    s.tone_presence[ch].set_high_shelf(static_cast<float>(kSampleRate), 6000.0f, (presence_raw - 0.5f) * 20.0f);
    if (bright_on)
      s.tight_boost[ch].set_high_pass(static_cast<float>(kSampleRate), 150.0f, 0.707f);
  }

  s.last_seen_knob_raw.store(model_select_raw, std::memory_order_release);
  bool ready = s.ready.load(std::memory_order_acquire);
  if (!ready)
  {
    switch_model_in_background(s, model_select_raw, s.force_recalibrate, false);
  }
  else
  {
    const int target_idx = knob_value_to_index(model_select_raw, static_cast<int>(g_cached_models.size()), false);
    if (target_idx != s.candidate_index)
    {
      s.candidate_index = target_idx;
      s.candidate_since_ms = now_ms();
    }
    else if (target_idx != -2 && target_idx != s.active_index.load(std::memory_order_acquire)
             && now_ms() - s.candidate_since_ms >= kKnobSettleMs)
    {
      if (s.fade_state.load(std::memory_order_relaxed) == 0)
      {
        s.fade_state.store(1, std::memory_order_relaxed);
        s.fade_progress = 0;
      }
    }

    if (s.fade_state.load(std::memory_order_relaxed) == 0 && s.active_index.load(std::memory_order_acquire) >= 0
        && s.calibrated_epoch.load(std::memory_order_acquire) != g_topology_epoch.load(std::memory_order_acquire))
    {
      s.force_recalibrate = true;
      s.fade_state.store(1, std::memory_order_relaxed);
      s.fade_progress = 0;
    }
  }

  if (s.fade_state.load(std::memory_order_relaxed) == 3 && !s.pending_ready.load(std::memory_order_acquire))
    switch_model_in_background(s, model_select_raw, s.force_recalibrate, false);

  const int fs_for_pickup = s.fade_state.load(std::memory_order_relaxed);
  if ((fs_for_pickup == 0 || fs_for_pickup == 3) && s.pending_ready.load(std::memory_order_acquire))
  {
    const int current_idx = knob_value_to_index(model_select_raw, static_cast<int>(g_cached_models.size()), false);
    const int pending = s.pending_index;
    if (ready && pending != current_idx && current_idx != -2 && !s.force_recalibrate)
    {
      for (int i = 0; i < kMaxChannels; ++i)
        s.pending_dsp[i].reset();
      s.pending_ready.store(false, std::memory_order_release);
      s.fade_state.store(3, std::memory_order_relaxed);
      switch_model_in_background(s, model_select_raw, s.force_recalibrate, false);
    }
    else
    {
      for (int i = 0; i < kMaxChannels; ++i)
        s.dsp[i] = std::move(s.pending_dsp[i]);
      s.active_index.store(pending, std::memory_order_release);
      s.calibrated_epoch.store(g_topology_epoch.load(std::memory_order_acquire), std::memory_order_release);
      s.force_recalibrate = false;
      s.pending_ready.store(false, std::memory_order_release);
      s.ready.store(true, std::memory_order_release);
      ready = true;
      if (fs_for_pickup == 3)
      {
        s.fade_state.store(pending < 0 ? 0 : 2, std::memory_order_relaxed);
        s.fade_progress = 0;
      }
    }
  }

  // Bypass (0x95a5 != 0) or model not yet ready: dry passthrough.
  // Checked AFTER the background model-switch state machine above so models
  // continue preloading even while the block is bypassed.
  if (bypass_byte != 0 || !ready || !s.dsp[0])
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
      if (input[ch] && output[ch])
        std::memcpy(output[ch], input[ch], static_cast<size_t>(numFrames) * sizeof(float));
    return;
  }

  // Both Gain (input trim) and Master (output level) use trim_gain():
  // 0% = silence (0.0), 50% = unity (1.0 / 0 dB), 100% = +12 dB boost,
  // matching Anxiety OD's input/output gain staging at 50%.
  const float in_gain = trim_gain(gain_raw);
  const float master_gain = trim_gain(master_raw);

  // Pre-NAM conditioning: Input Gain + Tight Cut filter + Clamp
  for (uint32_t ch = 0; ch < numChannels; ++ch)
  {
    if (!input[ch] || !output[ch])
      continue;
    for (int32_t i = 0; i < numFrames; ++i)
    {
      float sample = input[ch][i] * in_gain;
      if (bright_on)
        sample = s.tight_boost[ch].process(sample);
      output[ch][i] = std::clamp(sample, -1.0f, 1.0f);
    }
  }

  const int fade_state = s.fade_state.load(std::memory_order_relaxed);
  const bool silent_active = s.active_index.load(std::memory_order_relaxed) < 0;

  if (fade_state == 0) // Steady state
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      if (silent_active)
      {
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
        continue;
      }
      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);
    }
  }
  else if (fade_state == 1) // Fading OUT old model
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      if (silent_active)
      {
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
        continue;
      }
      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);
      for (int32_t i = 0; i < numFrames; ++i)
      {
        const float g = 1.0f - raised_cosine_ramp(s.fade_progress + i, kFadeOutLenSamples);
        output[ch][i] *= g;
      }
    }
    s.fade_progress += numFrames;
    if (s.fade_progress >= kFadeOutLenSamples)
    {
      s.fade_progress = 0;
      const int current_idx = knob_value_to_index(model_select_raw, static_cast<int>(g_cached_models.size()), false);
      if (current_idx == s.active_index.load(std::memory_order_acquire) && !s.force_recalibrate)
      {
        s.fade_state.store(s.active_index.load(std::memory_order_relaxed) < 0 ? 0 : 2, std::memory_order_relaxed);
      }
      else
      {
        s.fade_state.store(3, std::memory_order_relaxed);
        switch_model_in_background(s, model_select_raw, s.force_recalibrate, false);
      }
    }
  }
  else if (fade_state == 3) // Muted wait while background thread loads
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
      if (input[ch] && output[ch])
        std::fill(output[ch], output[ch] + numFrames, 0.0f);
  }
  else // fade_state == 2: Fading IN new model
  {
    for (uint32_t ch = 0; ch < numChannels; ++ch)
    {
      if (!input[ch] || !output[ch])
        continue;
      for (int32_t i = 0; i < numFrames; ++i)
        output[ch][i] *= raised_cosine_ramp(s.fade_progress + i, kFadeInLenSamples);

      float* in_arr[1] = {output[ch]};
      float* out_arr[1] = {output[ch]};
      s.dsp[ch]->process(in_arr, out_arr, numFrames);

      for (int32_t i = 0; i < numFrames; ++i)
      {
        const float g = raised_cosine_ramp(s.fade_progress + i, kFadeInLenSamples);
        output[ch][i] *= g;
      }
    }
    s.fade_progress += numFrames;
    if (s.fade_progress >= kFadeInLenSamples)
    {
      s.fade_progress = 0;
      s.fade_state.store(0, std::memory_order_relaxed);
    }
  }

  // Post-NAM conditioning: Active 3-Band EQ + Presence + Master Volume + Clamp
  for (uint32_t ch = 0; ch < numChannels; ++ch)
  {
    if (!input[ch] || !output[ch])
      continue;
    for (int32_t i = 0; i < numFrames; ++i)
    {
      float sample = output[ch][i];
      sample = s.tone_bass[ch].process(sample);
      sample = s.tone_mid[ch].process(sample);
      sample = s.tone_treble[ch].process(sample);
      sample = s.tone_presence[ch].process(sample);
      output[ch][i] = std::clamp(sample * master_gain, -1.0f, 1.0f);
    }
  }
}
