//! The 123 ROM waveforms, from the effect table in SLOS854D section 12.1.2.
//!
//! The discriminant is the waveform identifier the sequencer registers take, so a
//! variant casts straight to its slot byte. Percentages in the names are the amplitude
//! the effect plays at, and the two-number transition names are the start and end
//! amplitude of the ramp.

/// A ROM effect. The feel of each one depends on the selected library; this crate
/// selects ERM Library B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LibraryEffect {
    /// Strong Click - 100%
    StrongClickOneHundredPercent = 1,
    /// Strong Click - 60%
    StrongClickSixtyPercent = 2,
    /// Strong Click - 30%
    StrongClickThirtyPercent = 3,
    /// Sharp Click - 100%
    SharpClickOneHundredPercent = 4,
    /// Sharp Click - 60%
    SharpClickSixtyPercent = 5,
    /// Sharp Click - 30%
    SharpClickThirtyPercent = 6,
    /// Soft Bump - 100%
    SoftBumpOneHundredPercent = 7,
    /// Soft Bump - 60%
    SoftBumpSixtyPercent = 8,
    /// Soft Bump - 30%
    SoftBumpThirtyPercent = 9,
    /// Double Click - 100%
    DoubleClickOneHundredPercent = 10,
    /// Double Click - 60%
    DoubleClickSixtyPercent = 11,
    /// Triple Click - 100%
    TripleClickOneHundredPercent = 12,
    /// Soft Fuzz - 60%
    SoftFuzzSixtyPercent = 13,
    /// Strong Buzz - 100%
    StrongBuzzOneHundredPercent = 14,
    /// 750 ms Alert 100%
    SevenHundredFiftyMillisecondsAlertOneHundredPercent = 15,
    /// 1000 ms Alert 100%
    OneThousandMillisecondsAlertOneHundredPercent = 16,
    /// Strong Click 1 - 100%
    StrongClickOneOneHundredPercent = 17,
    /// Strong Click 2 - 80%
    StrongClickTwoEightyPercent = 18,
    /// Strong Click 3 - 60%
    StrongClickThreeSixtyPercent = 19,
    /// Strong Click 4 - 30%
    StrongClickFourThirtyPercent = 20,
    /// Medium Click 1 - 100%
    MediumClickOneOneHundredPercent = 21,
    /// Medium Click 2 - 80%
    MediumClickTwoEightyPercent = 22,
    /// Medium Click 3 - 60%
    MediumClickThreeSixtyPercent = 23,
    /// Sharp Tick 1 - 100%
    SharpTickOneOneHundredPercent = 24,
    /// Sharp Tick 2 - 80%
    SharpTickTwoEightyPercent = 25,
    /// Sharp Tick 3 - 60%
    SharpTickThreeSixtyPercent = 26,
    /// Short Double Click Strong 1 - 100%
    ShortDoubleClickStrongOneOneHundredPercent = 27,
    /// Short Double Click Strong 2 - 80%
    ShortDoubleClickStrongTwoEightyPercent = 28,
    /// Short Double Click Strong 3 - 60%
    ShortDoubleClickStrongThreeSixtyPercent = 29,
    /// Short Double Click Strong 4 - 30%
    ShortDoubleClickStrongFourThirtyPercent = 30,
    /// Short Double Click Medium 1 - 100%
    ShortDoubleClickMediumOneOneHundredPercent = 31,
    /// Short Double Click Medium 2 - 80%
    ShortDoubleClickMediumTwoEightyPercent = 32,
    /// Short Double Click Medium 3 - 60%
    ShortDoubleClickMediumThreeSixtyPercent = 33,
    /// Short Double Sharp Tick 1 - 100%
    ShortDoubleSharpTickOneOneHundredPercent = 34,
    /// Short Double Sharp Tick 2 - 80%
    ShortDoubleSharpTickTwoEightyPercent = 35,
    /// Short Double Sharp Tick 3 - 60%
    ShortDoubleSharpTickThreeSixtyPercent = 36,
    /// Long Double Sharp Click Strong 1 - 100%
    LongDoubleSharpClickStrongOneOneHundredPercent = 37,
    /// Long Double Sharp Click Strong 2 - 80%
    LongDoubleSharpClickStrongTwoEightyPercent = 38,
    /// Long Double Sharp Click Strong 3 - 60%
    LongDoubleSharpClickStrongThreeSixtyPercent = 39,
    /// Long Double Sharp Click Strong 4 - 30%
    LongDoubleSharpClickStrongFourThirtyPercent = 40,
    /// Long Double Sharp Click Medium 1 - 100%
    LongDoubleSharpClickMediumOneOneHundredPercent = 41,
    /// Long Double Sharp Click Medium 2 - 80%
    LongDoubleSharpClickMediumTwoEightyPercent = 42,
    /// Long Double Sharp Click Medium 3 - 60%
    LongDoubleSharpClickMediumThreeSixtyPercent = 43,
    /// Long Double Sharp Tick 1 - 100%
    LongDoubleSharpTickOneOneHundredPercent = 44,
    /// Long Double Sharp Tick 2 - 80%
    LongDoubleSharpTickTwoEightyPercent = 45,
    /// Long Double Sharp Tick 3 - 60%
    LongDoubleSharpTickThreeSixtyPercent = 46,
    /// Buzz 1 - 100%
    BuzzOneOneHundredPercent = 47,
    /// Buzz 2 - 80%
    BuzzTwoEightyPercent = 48,
    /// Buzz 3 - 60%
    BuzzThreeSixtyPercent = 49,
    /// Buzz 4 - 40%
    BuzzFourFortyPercent = 50,
    /// Buzz 5 - 20%
    BuzzFiveTwentyPercent = 51,
    /// Pulsing Strong 1 - 100%
    PulsingStrongOneOneHundredPercent = 52,
    /// Pulsing Strong 2 - 60%
    PulsingStrongTwoSixtyPercent = 53,
    /// Pulsing Medium 1 - 100%
    PulsingMediumOneOneHundredPercent = 54,
    /// Pulsing Medium 2 - 60%
    PulsingMediumTwoSixtyPercent = 55,
    /// Pulsing Sharp 1 - 100%
    PulsingSharpOneOneHundredPercent = 56,
    /// Pulsing Sharp 2 - 60%
    PulsingSharpTwoSixtyPercent = 57,
    /// Transition Click 1 - 100%
    TransitionClickOneOneHundredPercent = 58,
    /// Transition Click 2 - 80%
    TransitionClickTwoEightyPercent = 59,
    /// Transition Click 3 - 60%
    TransitionClickThreeSixtyPercent = 60,
    /// Transition Click 4 - 40%
    TransitionClickFourFortyPercent = 61,
    /// Transition Click 5 - 20%
    TransitionClickFiveTwentyPercent = 62,
    /// Transition Click 6 - 10%
    TransitionClickSixTenPercent = 63,
    /// Transition Hum 1 - 100%
    TransitionHumOneOneHundredPercent = 64,
    /// Transition Hum 2 - 80%
    TransitionHumTwoEightyPercent = 65,
    /// Transition Hum 3 - 60%
    TransitionHumThreeSixtyPercent = 66,
    /// Transition Hum 4 - 40%
    TransitionHumFourFortyPercent = 67,
    /// Transition Hum 5 - 20%
    TransitionHumFiveTwentyPercent = 68,
    /// Transition Hum 6 - 10%
    TransitionHumSixTenPercent = 69,
    /// Transition Ramp Down Long Smooth 1 - 100 to 0%
    TransitionRampDownLongSmoothOneOneHundredToZeroPercent = 70,
    /// Transition Ramp Down Long Smooth 2 - 100 to 0%
    TransitionRampDownLongSmoothTwoOneHundredToZeroPercent = 71,
    /// Transition Ramp Down Medium Smooth 1 - 100 to 0%
    TransitionRampDownMediumSmoothOneOneHundredToZeroPercent = 72,
    /// Transition Ramp Down Medium Smooth 2 - 100 to 0%
    TransitionRampDownMediumSmoothTwoOneHundredToZeroPercent = 73,
    /// Transition Ramp Down Short Smooth 1 - 100 to 0%
    TransitionRampDownShortSmoothOneOneHundredToZeroPercent = 74,
    /// Transition Ramp Down Short Smooth 2 - 100 to 0%
    TransitionRampDownShortSmoothTwoOneHundredToZeroPercent = 75,
    /// Transition Ramp Down Long Sharp 1 - 100 to 0%
    TransitionRampDownLongSharpOneOneHundredToZeroPercent = 76,
    /// Transition Ramp Down Long Sharp 2 - 100 to 0%
    TransitionRampDownLongSharpTwoOneHundredToZeroPercent = 77,
    /// Transition Ramp Down Medium Sharp 1 - 100 to 0%
    TransitionRampDownMediumSharpOneOneHundredToZeroPercent = 78,
    /// Transition Ramp Down Medium Sharp 2 - 100 to 0%
    TransitionRampDownMediumSharpTwoOneHundredToZeroPercent = 79,
    /// Transition Ramp Down Short Sharp 1 - 100 to 0%
    TransitionRampDownShortSharpOneOneHundredToZeroPercent = 80,
    /// Transition Ramp Down Short Sharp 2 - 100 to 0%
    TransitionRampDownShortSharpTwoOneHundredToZeroPercent = 81,
    /// Transition Ramp Up Long Smooth 1 - 0 to 100%
    TransitionRampUpLongSmoothOneZeroToOneHundredPercent = 82,
    /// Transition Ramp Up Long Smooth 2 - 0 to 100%
    TransitionRampUpLongSmoothTwoZeroToOneHundredPercent = 83,
    /// Transition Ramp Up Medium Smooth 1 - 0 to 100%
    TransitionRampUpMediumSmoothOneZeroToOneHundredPercent = 84,
    /// Transition Ramp Up Medium Smooth 2 - 0 to 100%
    TransitionRampUpMediumSmoothTwoZeroToOneHundredPercent = 85,
    /// Transition Ramp Up Short Smooth 1 - 0 to 100%
    TransitionRampUpShortSmoothOneZeroToOneHundredPercent = 86,
    /// Transition Ramp Up Short Smooth 2 - 0 to 100%
    TransitionRampUpShortSmoothTwoZeroToOneHundredPercent = 87,
    /// Transition Ramp Up Long Sharp 1 - 0 to 100%
    TransitionRampUpLongSharpOneZeroToOneHundredPercent = 88,
    /// Transition Ramp Up Long Sharp 2 - 0 to 100%
    TransitionRampUpLongSharpTwoZeroToOneHundredPercent = 89,
    /// Transition Ramp Up Medium Sharp 1 - 0 to 100%
    TransitionRampUpMediumSharpOneZeroToOneHundredPercent = 90,
    /// Transition Ramp Up Medium Sharp 2 - 0 to 100%
    TransitionRampUpMediumSharpTwoZeroToOneHundredPercent = 91,
    /// Transition Ramp Up Short Sharp 1 - 0 to 100%
    TransitionRampUpShortSharpOneZeroToOneHundredPercent = 92,
    /// Transition Ramp Up Short Sharp 2 - 0 to 100%
    TransitionRampUpShortSharpTwoZeroToOneHundredPercent = 93,
    /// Transition Ramp Down Long Smooth 1 - 50 to 0%
    TransitionRampDownLongSmoothOneFiftyToZeroPercent = 94,
    /// Transition Ramp Down Long Smooth 2 - 50 to 0%
    TransitionRampDownLongSmoothTwoFiftyToZeroPercent = 95,
    /// Transition Ramp Down Medium Smooth 1 - 50 to 0%
    TransitionRampDownMediumSmoothOneFiftyToZeroPercent = 96,
    /// Transition Ramp Down Medium Smooth 2 - 50 to 0%
    TransitionRampDownMediumSmoothTwoFiftyToZeroPercent = 97,
    /// Transition Ramp Down Short Smooth 1 - 50 to 0%
    TransitionRampDownShortSmoothOneFiftyToZeroPercent = 98,
    /// Transition Ramp Down Short Smooth 2 - 50 to 0%
    TransitionRampDownShortSmoothTwoFiftyToZeroPercent = 99,
    /// Transition Ramp Down Long Sharp 1 - 50 to 0%
    TransitionRampDownLongSharpOneFiftyToZeroPercent = 100,
    /// Transition Ramp Down Long Sharp 2 - 50 to 0%
    TransitionRampDownLongSharpTwoFiftyToZeroPercent = 101,
    /// Transition Ramp Down Medium Sharp 1 - 50 to 0%
    TransitionRampDownMediumSharpOneFiftyToZeroPercent = 102,
    /// Transition Ramp Down Medium Sharp 2 - 50 to 0%
    TransitionRampDownMediumSharpTwoFiftyToZeroPercent = 103,
    /// Transition Ramp Down Short Sharp 1 - 50 to 0%
    TransitionRampDownShortSharpOneFiftyToZeroPercent = 104,
    /// Transition Ramp Down Short Sharp 2 - 50 to 0%
    TransitionRampDownShortSharpTwoFiftyToZeroPercent = 105,
    /// Transition Ramp Up Long Smooth 1 - 0 to 50%
    TransitionRampUpLongSmoothOneZeroToFiftyPercent = 106,
    /// Transition Ramp Up Long Smooth 2 - 0 to 50%
    TransitionRampUpLongSmoothTwoZeroToFiftyPercent = 107,
    /// Transition Ramp Up Medium Smooth 1 - 0 to 50%
    TransitionRampUpMediumSmoothOneZeroToFiftyPercent = 108,
    /// Transition Ramp Up Medium Smooth 2 - 0 to 50%
    TransitionRampUpMediumSmoothTwoZeroToFiftyPercent = 109,
    /// Transition Ramp Up Short Smooth 1 - 0 to 50%
    TransitionRampUpShortSmoothOneZeroToFiftyPercent = 110,
    /// Transition Ramp Up Short Smooth 2 - 0 to 50%
    TransitionRampUpShortSmoothTwoZeroToFiftyPercent = 111,
    /// Transition Ramp Up Long Sharp 1 - 0 to 50%
    TransitionRampUpLongSharpOneZeroToFiftyPercent = 112,
    /// Transition Ramp Up Long Sharp 2 - 0 to 50%
    TransitionRampUpLongSharpTwoZeroToFiftyPercent = 113,
    /// Transition Ramp Up Medium Sharp 1 - 0 to 50%
    TransitionRampUpMediumSharpOneZeroToFiftyPercent = 114,
    /// Transition Ramp Up Medium Sharp 2 - 0 to 50%
    TransitionRampUpMediumSharpTwoZeroToFiftyPercent = 115,
    /// Transition Ramp Up Short Sharp 1 - 0 to 50%
    TransitionRampUpShortSharpOneZeroToFiftyPercent = 116,
    /// Transition Ramp Up Short Sharp 2 - 0 to 50%
    TransitionRampUpShortSharpTwoZeroToFiftyPercent = 117,
    /// Long buzz for programmatic stopping - 100%
    LongBuzzForProgrammaticStoppingOneHundredPercent = 118,
    /// Smooth Hum 1 (No kick or brake pulse) - 50%
    SmoothHumOneNoKickOrBrakePulseFiftyPercent = 119,
    /// Smooth Hum 2 (No kick or brake pulse) - 40%
    SmoothHumTwoNoKickOrBrakePulseFortyPercent = 120,
    /// Smooth Hum 3 (No kick or brake pulse) - 30%
    SmoothHumThreeNoKickOrBrakePulseThirtyPercent = 121,
    /// Smooth Hum 4 (No kick or brake pulse) - 20%
    SmoothHumFourNoKickOrBrakePulseTwentyPercent = 122,
    /// Smooth Hum 5 (No kick or brake pulse) - 10%
    SmoothHumFiveNoKickOrBrakePulseTenPercent = 123,
}
