/* The float32 kernels the firmware will mirror, at C speed.
 *
 * Every intermediate is a float, contraction is disabled at the build line, so
 * the arithmetic here is the plain single-precision sequence a Rust f32 loop
 * would produce. The Python module keeps a numpy transcription of the same
 * loops and checks the two agree bit for bit. */

#include <math.h>
#include <stddef.h>

/* One biquad, direct form II transposed, over one channel in place. */
void biquad_direct_form_two_transposed(float *signal, size_t count,
                                       float b0, float b1, float b2,
                                       float a1, float a2)
{
    float state_one = 0.0f;
    float state_two = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float sample = signal[index];
        float value = b0 * sample + state_one;
        state_one = b1 * sample - a1 * value + state_two;
        state_two = b2 * sample - a2 * value;
        signal[index] = value;
    }
}

/* A cascade of `sections` biquads, coefficients packed as b0,b1,b2,a1,a2. */
void biquad_cascade(float *signal, size_t count,
                    const float *coefficients, size_t sections)
{
    for (size_t section = 0; section < sections; section++) {
        const float *c = coefficients + 5 * section;
        biquad_direct_form_two_transposed(signal, count, c[0], c[1], c[2],
                                          c[3], c[4]);
    }
}

/* Mean power per quarter, log10, then the mean of the four logarithms.
 * `band` is one channel of one band; `out` receives one feature. */
float window_feature(const float *band, size_t sub_windows,
                     size_t sub_window_samples)
{
    float total = 0.0f;
    for (size_t quarter = 0; quarter < sub_windows; quarter++) {
        const float *segment = band + quarter * sub_window_samples;
        float accumulator = 0.0f;
        for (size_t index = 0; index < sub_window_samples; index++) {
            accumulator += segment[index] * segment[index];
        }
        float power = accumulator / (float)sub_window_samples;
        total += log10f(power + 1e-12f);
    }
    return total / (float)sub_windows;
}

/* Every channel of one band at one window start: `band` is (channels, samples)
 * row-major, `out` receives one feature per channel. */
void band_window_features(const float *band, size_t channels, size_t samples,
                          size_t at, size_t sub_windows,
                          size_t sub_window_samples, float *out)
{
    for (size_t channel = 0; channel < channels; channel++) {
        out[channel] = window_feature(band + channel * samples + at,
                                      sub_windows, sub_window_samples);
    }
}
