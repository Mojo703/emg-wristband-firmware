"""Design the notch and bandpass coefficients in float64 and freeze them as float32.

The device and the host simulation must filter with bit-identical coefficients.
Designing them twice invites two different roundings, so they are designed once
here, rounded once to float32, and written to fixtures/filter_coefficients.json
with the exact bit patterns alongside the decimal values. The firmware bakes the
bit patterns in through f32::from_bits, which also keeps them out of the Xtensa
float constant pool.

Every biquad is stored normalised to a0 = 1 and packed as b0, b1, b2, a1, a2 --
the order the direct-form II transposed update consumes them in.

    python3 generate_filter_coefficients.py            # write the fixture
    python3 generate_filter_coefficients.py --check     # verify it is current
    python3 generate_filter_coefficients.py --rust      # print the Rust tables
"""

import argparse
import json
import struct

import numpy as np

from bench_sessions import FIXTURES, SAMPLE_RATE
from reference_pipeline import band_sections, notch_sections
from score_requirements import MAINS_HARMONICS
from verify_front_end_fix import EMG_BAND

DESTINATION = FIXTURES / "filter_coefficients.json"
NOTCH_QUALITY = 30.0


def bit_pattern(value):
    """The exact float32 bit pattern of one already-rounded coefficient."""
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def packed_section(b0, b1, b2, a1, a2):
    """One biquad as decimals and bit patterns, both float32-exact."""
    values = [np.float32(coefficient) for coefficient in (b0, b1, b2, a1, a2)]
    return {
        "decimal": [float(value) for value in values],
        "bits": [bit_pattern(value) for value in values],
    }


def notch_rows():
    """The seven mains notches, in application order, normalised to a0 = 1."""
    rows = []
    for harmonic, (b, a) in zip(MAINS_HARMONICS, notch_sections()):
        section = packed_section(b[0] / a[0], b[1] / a[0], b[2] / a[0],
                                 a[1] / a[0], a[2] / a[0])
        section["harmonic_hz"] = float(harmonic)
        rows.append(section)
    return rows


def band_rows():
    """The four bandpass banks; butter's sos output already has a0 = 1."""
    banks = []
    for band, sections in zip(EMG_BAND, band_sections()):
        banks.append({
            "low_hz": float(band[0]),
            "high_hz": float(band[1]),
            "sections": [packed_section(row[0], row[1], row[2], row[4], row[5])
                         for row in sections],
        })
    return banks


def build():
    return {
        "sample_rate_hz": float(SAMPLE_RATE),
        "notch_quality": NOTCH_QUALITY,
        "notches": notch_rows(),
        "bands": band_rows(),
    }


def rust_tables(document):
    """The generated coefficient tables as Rust source, for band_features.rs."""
    lines = []

    def section_literal(section, indent):
        words = ", ".join(f"0x{bits:08x}" for bits in section["bits"])
        lines.append(f"{indent}[{words}],")

    lines.append("/// Seven mains notches (60..420 Hz, Q = 30) as f32 bit patterns,")
    lines.append("/// packed b0, b1, b2, a1, a2 with a0 normalized to 1.")
    lines.append("const NOTCH_BITS: [[u32; 5]; "
                 f"{len(document['notches'])}] = [")
    for section in document["notches"]:
        section_literal(section, "    ")
    lines.append("];")
    lines.append("")
    band_count = len(document["bands"])
    section_count = len(document["bands"][0]["sections"])
    edges = ", ".join(f"{band['low_hz']:.0f}-{band['high_hz']:.0f}"
                      for band in document["bands"])
    lines.append(f"/// The four Butterworth bandpasses ({edges} Hz), each an")
    lines.append("/// order-4 band design, which scipy returns as "
                 f"{section_count} second-order sections.")
    lines.append(f"const BAND_BITS: [[[u32; 5]; {section_count}]; {band_count}] = [")
    for band in document["bands"]:
        lines.append("    [")
        for section in band["sections"]:
            section_literal(section, "        ")
        lines.append("    ],")
    lines.append("];")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed fixture differs")
    parser.add_argument("--rust", action="store_true",
                        help="print the Rust coefficient tables and exit")
    arguments = parser.parse_args()

    document = build()
    if arguments.rust:
        print(rust_tables(document))
        return

    serialized = json.dumps(document, indent=2) + "\n"
    if arguments.check:
        if not DESTINATION.exists():
            raise SystemExit(f"{DESTINATION} does not exist")
        if DESTINATION.read_text() != serialized:
            raise SystemExit(f"{DESTINATION} is not what this script generates")
        print(f"{DESTINATION} is current")
        return

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(serialized)
    sections = sum(len(band["sections"]) for band in document["bands"])
    print(f"wrote {DESTINATION}: {len(document['notches'])} notches, "
          f"{len(document['bands'])} bands, {sections} bandpass sections")


if __name__ == "__main__":
    main()
