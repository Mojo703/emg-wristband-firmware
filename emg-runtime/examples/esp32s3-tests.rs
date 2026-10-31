//! ESP32-S3 checks for `emg-runtime`'s Xtensa SIMD inference path.
//!
//! These tests must run on the device. Off target, the SIMD entry points resolve
//! to scalar fallbacks and their comparisons prove nothing.

#[cfg(all(test, target_arch = "xtensa", target_os = "espidf"))]
mod tests {
    use std::hint::black_box;
    use std::io::Write;
    use std::time::Instant;

    use emg_runtime::band_features::{
        BandFeaturePipeline, FilterLayout, CHANNEL_COUNT, FEATURE_COUNT as BAND_FEATURE_COUNT,
        WINDOW_SAMPLES,
    };
    use emg_runtime::covariance_features::{
        CovarianceFeaturePipeline, CovarianceVariant, FEATURE_COUNT,
        SECTIONS_AFTER_INTEGER_SECOND_DIFFERENCE_BITS,
    };
    use emg_runtime::model::{ForwardResult, Model, VerifyBatch};
    use emg_runtime::tensor::{AlignedI8, I8Activation, Rng};
    use emg_runtime::{layers, mac};
    use esp_idf_svc as _;

    #[repr(align(16))]
    struct AlignedBlob<Bytes: ?Sized>(Bytes);

    static FIXTURE_BLOB: &AlignedBlob<[u8]> =
        &AlignedBlob(*include_bytes!("../data/model_int8_verify.bin"));

    /// Integer ratios avoid float constants because the Xtensa backend
    /// mishandles their constant pools in the test profile.
    const TOP1_FLOOR_PERCENT: usize = 85;
    const AGREEMENT_FLOOR_PERCENT: usize = 90;

    fn predicted_class(logits: &[i32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn reference_class(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    #[test]
    fn simd_dot_product_matches_scalar_oracle() {
        for &length in &[16usize, 32, 64, 128, 256] {
            let mut rng = Rng::new(0xC0DE + length as u32);
            let weights = AlignedI8::from_slice(&rng.fill_i8(length));
            let activations = AlignedI8::from_slice(&rng.fill_i8(length));

            assert_eq!(
                mac::dot_i8_scalar(weights.as_slice(), activations.as_slice()),
                mac::dot_i8_simd(weights.as_slice(), activations.as_slice()),
                "length {length}"
            );
        }
    }

    #[test]
    fn simd_depthwise_matches_scalar_oracle() {
        let kernel = 25;
        let requantize = layers::Requantize {
            mult: 1,
            shift: 12,
            relu: true,
        };

        for &(steps, channels) in &[(32usize, 16usize), (64, 32), (125, 64), (250, 16)] {
            let mut rng = Rng::new(0xDEAD + steps as u32);
            let input = I8Activation::synthetic(steps, channels, 0xBEEF + steps as u32);
            let weights = AlignedI8::from_slice(&rng.fill_i8(kernel * channels));
            let bias = rng.fill_i32_small(channels);

            let mut padded = I8Activation::zeros(1, 1);
            let mut scalar_output = I8Activation::zeros(1, 1);
            let mut simd_output = I8Activation::zeros(1, 1);

            let layer = layers::DepthwiseLayer {
                weights: weights.as_slice(),
                bias: &bias,
                kernel,
                stride: 2,
                requantize,
            };
            layers::depthwise_scalar(&input, layer, &mut padded, &mut scalar_output);
            layers::depthwise_simd(&input, layer, &mut padded, &mut simd_output);

            assert_eq!(
                scalar_output.as_slice(),
                simd_output.as_slice(),
                "steps {steps}, channels {channels}"
            );
        }
    }

    /// Top-1 checks model accuracy. Agreement checks that the int8 path still
    /// tracks the float path from which it was quantized.
    #[test]
    fn device_forward_pass_matches_float_reference() {
        let (mut model, _) = Model::load(
            &FIXTURE_BLOB.0,
            emg_runtime::model::ModelBuffers::reserve(&FIXTURE_BLOB.0),
        );
        let mut batch = VerifyBatch::new(&FIXTURE_BLOB.0);
        let total = batch.total;
        assert!(total > 0, "fixture blob carries no verification windows");

        let mut correct = 0usize;
        let mut agreements = 0usize;
        while let Some(window) = batch.next_window() {
            let ForwardResult::Logits(logits) = model.forward(&window.input);
            let predicted = predicted_class(&logits);
            if predicted == window.label as usize {
                correct += 1;
            }
            if predicted == reference_class(&window.float_logits) {
                agreements += 1;
            }
        }

        assert!(
            correct * 100 >= total * TOP1_FLOOR_PERCENT,
            "top-1 {correct}/{total}"
        );
        assert!(
            agreements * 100 >= total * AGREEMENT_FLOOR_PERCENT,
            "agreement {agreements}/{total}"
        );
    }

    const MICROVOLTS_PER_COUNT: f32 = 12.207031;
    const FIXTURE_WINDOWS: usize = 8;
    const FIXTURE_RAW_BYTES: usize = FIXTURE_WINDOWS * WINDOW_SAMPLES * CHANNEL_COUNT * 2;
    const WINDOW_BUDGET_MICROSECONDS: u64 = 250_000;
    const WORST_ERROR_TOLERANCE: f64 = 1e-2;
    const MEAN_ERROR_TOLERANCE: f64 = 1e-4;

    static COVARIANCE_FIXTURES: [(&str, &[u8]); 2] = [
        (
            "zachary",
            include_bytes!("../tests/fixtures/covariance_zachary_8_windows.bin"),
        ),
        (
            "matthew",
            include_bytes!("../tests/fixtures/covariance_matthew_8_windows.bin"),
        ),
    ];

    type Sample = [i16; CHANNEL_COUNT];

    fn fixture_sample(fixture: &[u8], index: usize) -> Sample {
        let bytes = &fixture[index * CHANNEL_COUNT * 2..(index + 1) * CHANNEL_COUNT * 2];
        let mut sample = [0i16; CHANNEL_COUNT];
        for (value, pair) in sample.iter_mut().zip(bytes.chunks_exact(2)) {
            *value = i16::from_le_bytes([pair[0], pair[1]]);
        }
        sample
    }

    fn fixture_reference(fixture: &[u8], index: usize) -> f64 {
        let start = FIXTURE_RAW_BYTES + index * 8;
        f64::from_le_bytes(fixture[start..start + 8].try_into().expect("eight bytes"))
    }

    fn report(line: std::fmt::Arguments) {
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_fmt(line);
        let _ = stdout.write_all(b"\n");
        let _ = stdout.flush();
    }

    fn hundredths(numerator: u64, denominator: u64) -> String {
        let scaled = numerator * 10_000 / denominator;
        format!("{}.{:02}", scaled / 100, scaled % 100)
    }

    #[derive(Default)]
    struct WindowTimings {
        windows: u64,
        streaming_microseconds: u64,
        closing_microseconds: u64,
        worst_window_microseconds: u64,
    }

    impl WindowTimings {
        fn record(&mut self, streaming: u64, closing: u64) {
            self.windows += 1;
            self.streaming_microseconds += streaming;
            self.closing_microseconds += closing;
            self.worst_window_microseconds =
                self.worst_window_microseconds.max(streaming + closing);
        }

        fn summary(&self) -> String {
            let streamed_samples = self.windows * (WINDOW_SAMPLES as u64 - 1);
            let nanoseconds_per_sample = self.streaming_microseconds * 1000 / streamed_samples;
            let window_finish = (self.closing_microseconds * 1000)
                .saturating_sub(self.windows * nanoseconds_per_sample)
                / self.windows
                / 1000;
            let mean_window =
                (self.streaming_microseconds + self.closing_microseconds) / self.windows;
            format!(
                "push {}.{:02} us/sample, finish {window_finish} us, window {}.{:02} ms mean ({}%), {}.{:02} ms worst ({}%)",
                nanoseconds_per_sample / 1000,
                nanoseconds_per_sample % 1000 / 10,
                mean_window / 1000,
                mean_window % 1000 / 10,
                hundredths(mean_window, WINDOW_BUDGET_MICROSECONDS),
                self.worst_window_microseconds / 1000,
                self.worst_window_microseconds % 1000 / 10,
                hundredths(self.worst_window_microseconds, WINDOW_BUDGET_MICROSECONDS),
            )
        }
    }

    fn free_heap() -> u32 {
        // SAFETY: esp_get_free_heap_size only reads allocator statistics.
        unsafe { esp_idf_svc::sys::esp_get_free_heap_size() }
    }

    fn time_window(
        push: &mut impl FnMut(&Sample) -> bool,
        fixture: &[u8],
        window: usize,
        window_buffer: &mut [Sample],
        timings: &mut WindowTimings,
    ) {
        for (offset, sample) in window_buffer.iter_mut().enumerate() {
            *sample = fixture_sample(fixture, window * WINDOW_SAMPLES + offset);
        }
        let (closing_sample, streamed) = window_buffer.split_last().expect("window has samples");
        let started = Instant::now();
        for sample in streamed {
            assert!(!push(sample));
        }
        let streamed_at = Instant::now();
        assert!(push(closing_sample));
        let closed_at = Instant::now();
        timings.record(
            (streamed_at - started).as_micros() as u64,
            (closed_at - streamed_at).as_micros() as u64,
        );
    }

    const BENCHMARK_STACK_BYTES: usize = 48 * 1024;

    fn on_large_stack(body: fn()) {
        std::thread::Builder::new()
            .stack_size(BENCHMARK_STACK_BYTES)
            .spawn(body)
            .expect("benchmark thread starts")
            .join()
            .expect("benchmark thread succeeds");
    }

    #[test]
    fn covariance_variant_timings_parity_and_memory() {
        on_large_stack(covariance_variant_benchmark);
    }

    #[test]
    fn band_feature_layout_timings_and_agreement() {
        on_large_stack(band_feature_layout_benchmark);
    }

    fn covariance_variant_benchmark() {
        let mut window_buffer = vec![[0i16; CHANNEL_COUNT]; WINDOW_SAMPLES];
        for (label, variant) in CovarianceVariant::NAMED {
            let mut timings = WindowTimings::default();
            let mut heap_bytes = 0u32;
            let mut worst = 0.0f64;
            let mut total_error = 0.0f64;
            let mut sweeps = 0u64;
            let mut feature_hash = 0xcbf2_9ce4_8422_2325u64;
            for (_, fixture) in COVARIANCE_FIXTURES {
                let free_before = free_heap();
                let mut pipeline =
                    CovarianceFeaturePipeline::with_variant(MICROVOLTS_PER_COUNT, variant);
                heap_bytes = free_before - free_heap();
                let latest = std::cell::Cell::new([0.0f32; FEATURE_COUNT]);
                let mut push = |sample: &Sample| match black_box(pipeline.push(black_box(sample))) {
                    Some(features) => {
                        latest.set(*features);
                        true
                    }
                    None => false,
                };
                for window in 0..FIXTURE_WINDOWS {
                    time_window(&mut push, fixture, window, &mut window_buffer, &mut timings);
                    for (entry, &value) in latest.get().iter().enumerate() {
                        let expected = fixture_reference(fixture, window * FEATURE_COUNT + entry);
                        let error = (value as f64 - expected).abs();
                        worst = worst.max(error);
                        total_error += error;
                        feature_hash = (feature_hash ^ u64::from(value.to_bits()))
                            .wrapping_mul(0x0100_0000_01b3);
                    }
                }
                drop(push);
                sweeps += pipeline
                    .last_window_sweeps()
                    .iter()
                    .map(|&count| u64::from(count))
                    .sum::<u64>();
            }
            let mean = total_error / (2 * FIXTURE_WINDOWS * FEATURE_COUNT) as f64;
            report(format_args!(
                "covariance {label}: {}; heap {heap_bytes} B; parity mean {mean:.2e} worst {worst:.2e}; last-window sweeps {sweeps}; hash {feature_hash:016x}",
                timings.summary(),
            ));
            if variant == CovarianceVariant::BASELINE {
                assert!(worst <= WORST_ERROR_TOLERANCE, "worst error {worst:e}");
                assert!(mean <= MEAN_ERROR_TOLERANCE, "mean error {mean:e}");
                assert!(timings.worst_window_microseconds < WINDOW_BUDGET_MICROSECONDS);
            }
        }
    }

    #[test]
    fn esp_dsp_biquad_against_rust_biquads() {
        on_large_stack(esp_dsp_biquad_benchmark);
    }

    extern "C" {
        fn dsps_biquad_f32_aes3(
            input: *const f32,
            output: *mut f32,
            length: i32,
            coefficients: *mut f32,
            state: *mut f32,
        ) -> i32;
    }

    type SectionCoefficients = [[[f32; 5]; 4]; 4];

    fn covariance_sections() -> SectionCoefficients {
        SECTIONS_AFTER_INTEGER_SECOND_DIFFERENCE_BITS
            .map(|band| band.map(|section| section.map(f32::from_bits)))
    }

    fn second_difference_window(
        fixture: &[u8],
        window: usize,
        previous: &mut [[i32; 2]; CHANNEL_COUNT],
        channels: &mut [Vec<f32>],
    ) {
        for offset in 0..WINDOW_SAMPLES {
            let sample = fixture_sample(fixture, window * WINDOW_SAMPLES + offset);
            for (channel, &count) in sample.iter().enumerate() {
                let count = i32::from(count);
                let difference = count - 2 * previous[channel][0] + previous[channel][1];
                previous[channel] = [count, previous[channel][0]];
                channels[channel][offset] = difference as f32 * MICROVOLTS_PER_COUNT;
            }
        }
    }

    fn transposed_block(
        sections: &[[f32; 5]; 4],
        states: &mut [[f32; 2]; 4],
        input: &[f32],
        output: &mut [f32],
    ) {
        output.copy_from_slice(input);
        for (section, state) in sections.iter().zip(states.iter_mut()) {
            for value in output.iter_mut() {
                let sample = *value;
                let filtered = section[0] * sample + state[0];
                state[0] = section[1] * sample - section[3] * filtered + state[1];
                state[1] = section[2] * sample - section[4] * filtered;
                *value = filtered;
            }
        }
    }

    fn transposed_block_double(
        sections: &[[f32; 5]; 4],
        states: &mut [[f64; 2]; 4],
        input: &[f32],
        output: &mut [f64],
    ) {
        for (value, &sample) in output.iter_mut().zip(input.iter()) {
            *value = f64::from(sample);
        }
        for (section, state) in sections.iter().zip(states.iter_mut()) {
            let coefficient = section.map(f64::from);
            for value in output.iter_mut() {
                let sample = *value;
                let filtered = coefficient[0] * sample + state[0];
                state[0] = coefficient[1] * sample - coefficient[3] * filtered + state[1];
                state[1] = coefficient[2] * sample - coefficient[4] * filtered;
                *value = filtered;
            }
        }
    }

    fn esp_dsp_biquad_benchmark() {
        let sections = covariance_sections();
        let (_, fixture) = COVARIANCE_FIXTURES[0];
        let mut previous = [[0i32; 2]; CHANNEL_COUNT];
        let mut inputs = vec![vec![0.0f32; WINDOW_SAMPLES]; CHANNEL_COUNT];
        let mut output = vec![0.0f32; WINDOW_SAMPLES];
        let mut exact = vec![0.0f64; WINDOW_SAMPLES];
        let mut esp_dsp_states = [[[[0.0f32; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut esp_dsp_coefficients = sections;
        let mut transposed_states = [[[[0.0f32; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut exact_states = [[[[0.0f64; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut esp_dsp_microseconds = 0u64;
        let mut transposed_microseconds = 0u64;
        let mut esp_dsp_worst = 0.0f64;
        let mut transposed_worst = 0.0f64;
        let mut largest = 0.0f64;
        for window in 0..FIXTURE_WINDOWS {
            second_difference_window(fixture, window, &mut previous, &mut inputs);

            let started = Instant::now();
            for (channel, input) in inputs.iter().enumerate() {
                for band in 0..4 {
                    let mut source = input.as_ptr();
                    for section in 0..4 {
                        // SAFETY: input and output each hold WINDOW_SAMPLES floats; the
                        // coefficient and state arrays hold five and two floats as the
                        // esp-dsp biquad requires; esp-dsp reads each input before writing
                        // the matching output, so the in-place passes are sound.
                        unsafe {
                            dsps_biquad_f32_aes3(
                                source,
                                output.as_mut_ptr(),
                                WINDOW_SAMPLES as i32,
                                esp_dsp_coefficients[band][section].as_mut_ptr(),
                                esp_dsp_states[channel][band][section].as_mut_ptr(),
                            );
                        }
                        source = output.as_ptr();
                    }
                    black_box(&output);
                }
            }
            esp_dsp_microseconds += started.elapsed().as_micros() as u64;

            let started = Instant::now();
            for (channel, input) in inputs.iter().enumerate() {
                for band in 0..4 {
                    transposed_block(
                        &sections[band],
                        &mut transposed_states[channel][band],
                        input,
                        &mut output,
                    );
                    black_box(&output);
                }
            }
            transposed_microseconds += started.elapsed().as_micros() as u64;

            for (channel, input) in inputs.iter().enumerate() {
                for band in 0..4 {
                    transposed_block_double(
                        &sections[band],
                        &mut exact_states[channel][band],
                        input,
                        &mut exact,
                    );
                    largest = exact
                        .iter()
                        .fold(largest, |most, value| most.max(value.abs()));
                }
            }
        }
        let mut esp_dsp_states = [[[[0.0f32; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut transposed_states = [[[[0.0f32; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut exact_states = [[[[0.0f64; 2]; 4]; 4]; CHANNEL_COUNT];
        let mut previous = [[0i32; 2]; CHANNEL_COUNT];
        let mut esp_dsp_output = vec![0.0f32; WINDOW_SAMPLES];
        for window in 0..FIXTURE_WINDOWS {
            second_difference_window(fixture, window, &mut previous, &mut inputs);
            for (channel, input) in inputs.iter().enumerate() {
                for band in 0..4 {
                    let mut source = input.as_ptr();
                    for section in 0..4 {
                        // SAFETY: as in the timed loop above.
                        unsafe {
                            dsps_biquad_f32_aes3(
                                source,
                                esp_dsp_output.as_mut_ptr(),
                                WINDOW_SAMPLES as i32,
                                esp_dsp_coefficients[band][section].as_mut_ptr(),
                                esp_dsp_states[channel][band][section].as_mut_ptr(),
                            );
                        }
                        source = esp_dsp_output.as_ptr();
                    }
                    transposed_block(
                        &sections[band],
                        &mut transposed_states[channel][band],
                        input,
                        &mut output,
                    );
                    transposed_block_double(
                        &sections[band],
                        &mut exact_states[channel][band],
                        input,
                        &mut exact,
                    );
                    for ((&from_esp_dsp, &from_transposed), &reference) in
                        esp_dsp_output.iter().zip(output.iter()).zip(exact.iter())
                    {
                        esp_dsp_worst =
                            esp_dsp_worst.max((f64::from(from_esp_dsp) - reference).abs());
                        transposed_worst =
                            transposed_worst.max((f64::from(from_transposed) - reference).abs());
                    }
                }
            }
        }
        let samples = (FIXTURE_WINDOWS * WINDOW_SAMPLES) as u64;
        report(format_args!(
            "esp-dsp dsps_biquad_f32_aes3 block filter: {} ns/sample for 16 channels x 16 sections, worst error vs f64 {esp_dsp_worst:.2e} uV",
            esp_dsp_microseconds * 1000 / samples,
        ));
        report(format_args!(
            "rust transposed block filter: {} ns/sample for 16 channels x 16 sections, worst error vs f64 {transposed_worst:.2e} uV (largest output {largest:.3e} uV)",
            transposed_microseconds * 1000 / samples,
        ));
    }

    fn band_feature_layout_benchmark() {
        let mut window_buffer = vec![[0i16; CHANNEL_COUNT]; WINDOW_SAMPLES];
        let layouts = [
            ("channel-major", FilterLayout::ChannelMajor),
            ("section-major", FilterLayout::SectionMajor),
            ("section-major-fused", FilterLayout::SectionMajorFused),
        ];
        let mut baseline_features = vec![[0.0f32; BAND_FEATURE_COUNT]; 2 * FIXTURE_WINDOWS];
        for (label, layout) in layouts {
            let mut timings = WindowTimings::default();
            let mut differing = 0usize;
            let mut worst_difference = 0.0f32;
            for (fixture_index, (_, fixture)) in COVARIANCE_FIXTURES.iter().enumerate() {
                let mut pipeline =
                    BandFeaturePipeline::new(MICROVOLTS_PER_COUNT, [1.0; CHANNEL_COUNT])
                        .with_filter_layout(layout);
                let latest = std::cell::Cell::new([0.0f32; BAND_FEATURE_COUNT]);
                let mut push = |sample: &Sample| match black_box(pipeline.push(black_box(sample))) {
                    Some(features) => {
                        latest.set(features);
                        true
                    }
                    None => false,
                };
                for window in 0..FIXTURE_WINDOWS {
                    time_window(&mut push, fixture, window, &mut window_buffer, &mut timings);
                    let slot = &mut baseline_features[fixture_index * FIXTURE_WINDOWS + window];
                    if layout == FilterLayout::ChannelMajor {
                        *slot = latest.get();
                    }
                    for (&produced, &expected) in latest.get().iter().zip(slot.iter()) {
                        differing += usize::from(produced.to_bits() != expected.to_bits());
                        worst_difference = worst_difference.max((produced - expected).abs());
                    }
                }
            }
            report(format_args!(
                "band {label}: {}; inline {} B; {differing} features differ from channel-major, worst {worst_difference:.2e}",
                timings.summary(),
                core::mem::size_of::<BandFeaturePipeline>(),
            ));
            if layout == FilterLayout::SectionMajor {
                assert_eq!(differing, 0);
            }
        }
    }

    const FIT_ROWS: usize = 256;
    const FIT_CLASSES: usize = 3;
    const FIT_PASSES: usize = 4;

    #[test]
    fn calibration_fit_pass_timings() {
        on_large_stack(calibration_fit_benchmark);
    }

    /// The shipped fitter's pass at 64 features, then the same per-row
    /// arithmetic over wider rows, so a 544-input fit can be priced from the
    /// ratio at 64.
    fn calibration_fit_benchmark() {
        use emg_runtime::streaming_fit::{
            FitCheckpoint, Fitter, RowSource, StandardizedQuantization, ROW_STRIDE,
        };

        let mut rng = Rng::new(0xF17);
        let mut packed = vec![0u8; FIT_ROWS * ROW_STRIDE];
        for (index, row) in packed.chunks_exact_mut(ROW_STRIDE).enumerate() {
            let codes: [u8; BAND_FEATURE_COUNT] =
                core::array::from_fn(|feature| rng.fill_i8(1)[0] as u8 ^ feature as u8);
            RowSource::pack(&codes, (index % FIT_CLASSES) as u8, 1.0, row);
        }
        let quantization = StandardizedQuantization::uniform(10.0 / 127.0);
        let sources = [RowSource::new(&packed).expect("whole rows")];
        let mut fitter = Fitter::new(FIT_CLASSES);
        let mut checkpoint = FitCheckpoint::zeroed(FIT_CLASSES);
        let started = Instant::now();
        fitter.resume_fit(&mut checkpoint, &quantization, &sources, FIT_PASSES);
        let shipped = started.elapsed().as_nanos() as u64 / (FIT_ROWS * FIT_PASSES) as u64;
        report(format_args!(
            "fit: shipped fitter, 64 features x {FIT_CLASSES} classes: {shipped} ns per row-pass"
        ));

        for inputs in [64usize, 480, 544] {
            let stride = (inputs + 8).next_multiple_of(4);
            let rows: Vec<u8> = (0..FIT_ROWS * stride)
                .map(|index| rng.fill_i8(1)[0] as u8 ^ index as u8)
                .collect();
            let scale = vec![0.08f32; inputs];
            let offset = vec![0.0f32; inputs];
            let mut weights = vec![0.0f32; (inputs + 1) * FIT_CLASSES];
            let mut gradient = vec![0.0f32; (inputs + 1) * FIT_CLASSES];
            let mut design = vec![1.0f32; inputs + 1];
            let free_before = free_heap();
            let started = Instant::now();
            for _ in 0..FIT_PASSES {
                gradient.fill(0.0);
                for (index, row) in rows.chunks_exact(stride).enumerate() {
                    for ((input, &code), (&step, &shift)) in design[..inputs]
                        .iter_mut()
                        .zip(&row[..inputs])
                        .zip(scale.iter().zip(&offset))
                    {
                        *input = (code as i8) as f32 * step + shift;
                    }
                    let mut probabilities = [0.0f32; FIT_CLASSES];
                    for (&input, row_weights) in
                        design.iter().zip(weights.chunks_exact(FIT_CLASSES))
                    {
                        for (accumulator, &weight) in probabilities.iter_mut().zip(row_weights) {
                            *accumulator += input * weight;
                        }
                    }
                    let mut largest = probabilities[0];
                    for &value in &probabilities {
                        largest = largest.max(value);
                    }
                    let mut total = 0.0f32;
                    for value in probabilities.iter_mut() {
                        *value = libm::expf(*value - largest);
                        total += *value;
                    }
                    let inverse = 1.0 / total;
                    for value in probabilities.iter_mut() {
                        *value *= inverse;
                    }
                    probabilities[index % FIT_CLASSES] -= 1.0;
                    for (&input, accumulator) in
                        design.iter().zip(gradient.chunks_exact_mut(FIT_CLASSES))
                    {
                        for (slot, &residual) in accumulator.iter_mut().zip(&probabilities) {
                            *slot += input * residual;
                        }
                    }
                }
                let count = FIT_ROWS as f32;
                for (weight, &accumulated) in weights.iter_mut().zip(&gradient) {
                    *weight -= accumulated / count + *weight / 100.0;
                }
            }
            let nanoseconds = started.elapsed().as_nanos() as u64 / (FIT_ROWS * FIT_PASSES) as u64;
            black_box(&weights);
            report(format_args!(
                "fit: generic kernel, {inputs} inputs x {FIT_CLASSES} classes: {nanoseconds} ns per row-pass, {stride} B per row, free heap {free_before} B"
            ));
        }
    }
}

#[cfg(not(test))]
fn main() {
    panic!("run with `cargo +esp test-device`");
}
