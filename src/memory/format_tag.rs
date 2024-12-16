use onednnl_sys::dnnl_format_tag_t;

pub trait FormatTag<const NDIMS: usize> {
    /// The underlying oneDNN memory format tag.
    const TAG: dnnl_format_tag_t::Type;
}

#[macro_use]
mod macros {

    /// Macro to implement FormatTag trait for a given memory format tag.
    ///
    /// # Arguments
    ///
    /// * `$tag_struct` - The Rust struct name representing the memory format tag.
    /// * `$c_enum` - The corresponding C enum variant from `dnnl_memory_format_tag_t`.
    /// * `$ndims` - The number of dimensions the format tag supports.
    /// * `$comment` - A descriptive comment for the memory format tag.
    macro_rules! impl_format_tag {
        ($tag_struct:ident, $c_enum:ident, $ndims:expr, $comment:literal) => {
            #[doc = $comment]
            pub struct $tag_struct;

            impl FormatTag<$ndims> for $tag_struct {
                const TAG: dnnl_format_tag_t::Type = dnnl_format_tag_t::$c_enum;
            }
        };
    }

    // Make the macro available outside the module
    pub(crate) use impl_format_tag;
}

use macros::impl_format_tag;

impl_format_tag!(a, dnnl_a, 1, "< plain 1D tensor");
impl_format_tag!(ab, dnnl_ab, 2, "< plain 2D tensor");
impl_format_tag!(abc, dnnl_abc, 3, "< plain 3D tensor");
impl_format_tag!(abcd, dnnl_abcd, 4, "< plain 4D tensor");
impl_format_tag!(abcde, dnnl_abcde, 5, "< plain 5D tensor");
impl_format_tag!(abcdef, dnnl_abcdef, 6, "< plain 6D tensor");
impl_format_tag!(abcdefg, dnnl_abcdefg, 7, "< plain 7D tensor");
impl_format_tag!(abcdefgh, dnnl_abcdefgh, 8, "< plain 8D tensor");
impl_format_tag!(abcdefghi, dnnl_abcdefghi, 9, "< plain 9D tensor");
impl_format_tag!(abcdefghij, dnnl_abcdefghij, 10, "< plain 10D tensor");
impl_format_tag!(abcdefghijk, dnnl_abcdefghijk, 11, "< plain 11D tensor");
impl_format_tag!(abcdefghijkl, dnnl_abcdefghijkl, 12, "< plain 12D tensor");
impl_format_tag!(ba, dnnl_ba, 2, "< permuted 2D tensor");
impl_format_tag!(acb, dnnl_acb, 3, "< permuted 3D tensor");
impl_format_tag!(bac, dnnl_bac, 3, "< permuted 3D tensor");
impl_format_tag!(bca, dnnl_bca, 3, "< permuted 3D tensor");
impl_format_tag!(cab, dnnl_cab, 3, "< permuted 3D tensor");
impl_format_tag!(cba, dnnl_cba, 3, "< permuted 3D tensor");
impl_format_tag!(abdc, dnnl_abdc, 4, "< permuted 4D tensor");
impl_format_tag!(acbd, dnnl_acbd, 4, "< permuted 4D tensor");
impl_format_tag!(acdb, dnnl_acdb, 4, "< permuted 4D tensor");
impl_format_tag!(adbc, dnnl_adbc, 4, "< permuted 4D tensor");
impl_format_tag!(adcb, dnnl_adcb, 4, "< permuted 4D tensor");
impl_format_tag!(bacd, dnnl_bacd, 4, "< permuted 4D tensor");
impl_format_tag!(bcda, dnnl_bcda, 4, "< permuted 4D tensor");
impl_format_tag!(cdab, dnnl_cdab, 4, "< permuted 4D tensor");
impl_format_tag!(cdba, dnnl_cdba, 4, "< permuted 4D tensor");
impl_format_tag!(dcab, dnnl_dcab, 4, "< permuted 4D tensor");
impl_format_tag!(abced, dnnl_abced, 5, "< permuted 5D tensor");
impl_format_tag!(abdec, dnnl_abdec, 5, "< permuted 5D tensor");
impl_format_tag!(acbde, dnnl_acbde, 5, "< permuted 5D tensor");
impl_format_tag!(acdeb, dnnl_acdeb, 5, "< permuted 5D tensor");
impl_format_tag!(adecb, dnnl_adecb, 5, "< permuted 5D tensor");
impl_format_tag!(bacde, dnnl_bacde, 5, "< permuted 5D tensor");
impl_format_tag!(bcdea, dnnl_bcdea, 5, "< permuted 5D tensor");
impl_format_tag!(cdeab, dnnl_cdeab, 5, "< permuted 5D tensor");
impl_format_tag!(cdeba, dnnl_cdeba, 5, "< permuted 5D tensor");
impl_format_tag!(decab, dnnl_decab, 5, "< permuted 5D tensor");
impl_format_tag!(abcdfe, dnnl_abcdfe, 6, "< permuted 6D tensor");
impl_format_tag!(abdefc, dnnl_abdefc, 6, "< permuted 6D tensor");
impl_format_tag!(abdfce, dnnl_abdfce, 6, "< permuted 6D tensor");
impl_format_tag!(acbdef, dnnl_acbdef, 6, "< permuted 6D tensor");
impl_format_tag!(adefcb, dnnl_adefcb, 6, "< permuted 6D tensor");
impl_format_tag!(defcab, dnnl_defcab, 6, "< permuted 6D tensor");
impl_format_tag!(abcdegf, dnnl_abcdegf, 7, "< permuted 7D tensor");
impl_format_tag!(abcdefhg, dnnl_abcdefhg, 8, "< permuted 8D tensor");
impl_format_tag!(abcdefgih, dnnl_abcdefgih, 9, "< permuted 9D tensor");
impl_format_tag!(abcdefghji, dnnl_abcdefghji, 10, "< permuted 10D tensor");
impl_format_tag!(abcdefghikj, dnnl_abcdefghikj, 11, "< permuted 11D tensor");
impl_format_tag!(abcdefghijlk, dnnl_abcdefghijlk, 12, "< permuted 12D tensor");
impl_format_tag!(
    aBc16b,
    dnnl_aBc16b,
    3,
    "3D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    ABc16b16a,
    dnnl_ABc16b16a,
    3,
    "3D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    Abc4a,
    dnnl_Abc4a,
    3,
    "3D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBc32b,
    dnnl_aBc32b,
    3,
    "3D tensor blocked by 2nd dimension with block size 32"
);
impl_format_tag!(
    aBc4b,
    dnnl_aBc4b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b16a4b,
    dnnl_ABc4b16a4b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc2b8a4b,
    dnnl_ABc2b8a4b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b16a4b,
    dnnl_ABc16b16a4b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b16a2b,
    dnnl_ABc16b16a2b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b4a,
    dnnl_ABc4b4a,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8a16b2a,
    dnnl_ABc8a16b2a,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8a8b,
    dnnl_ABc8a8b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8a4b,
    dnnl_ABc8a4b,
    3,
    "3D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBc8b,
    dnnl_aBc8b,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABc8b16a2b,
    dnnl_ABc8b16a2b,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    BAc8a16b2a,
    dnnl_BAc8a16b2a,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABc8b8a,
    dnnl_ABc8b8a,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    Abcd16a,
    dnnl_Abcd16a,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    Abcd8a,
    dnnl_Abcd8a,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd16a16b,
    dnnl_ABcd16a16b,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    Abcd32a,
    dnnl_Abcd32a,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd32a32b,
    dnnl_ABcd32a32b,
    3,
    "3D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBcd16b,
    dnnl_aBcd16b,
    4,
    "4D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    ABcd16b16a,
    dnnl_ABcd16b16a,
    4,
    "4D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCd16b16c,
    dnnl_aBCd16b16c,
    4,
    "4D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCd16c16b,
    dnnl_aBCd16c16b,
    4,
    "4D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    Abcd4a,
    dnnl_Abcd4a,
    4,
    "4D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBcd32b,
    dnnl_aBcd32b,
    4,
    "4D tensor blocked by 2nd dimension with block size 32"
);
impl_format_tag!(
    aBcd4b,
    dnnl_aBcd4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b16a4b,
    dnnl_ABcd4b16a4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b16a4b,
    dnnl_ABcd16b16a4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b16a2b,
    dnnl_ABcd16b16a2b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b4a,
    dnnl_ABcd4b4a,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4a4b,
    dnnl_ABcd4a4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd2c4b2c,
    dnnl_aBCd2c4b2c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd4b8c2b,
    dnnl_aBCd4b8c2b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd4c16b4c,
    dnnl_aBCd4c16b4c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd2c8b4c,
    dnnl_aBCd2c8b4c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd16c16b4c,
    dnnl_aBCd16c16b4c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd16c16b2c,
    dnnl_aBCd16c16b2c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd4c4b,
    dnnl_aBCd4c4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd4b4c,
    dnnl_aBCd4b4c,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8a16b2a,
    dnnl_ABcd8a16b2a,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd2b8a4b,
    dnnl_ABcd2b8a4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8a8b,
    dnnl_ABcd8a8b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8a4b,
    dnnl_ABcd8a4b,
    4,
    "4D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBcd8b,
    dnnl_aBcd8b,
    4,
    "4D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd4c8b2c,
    dnnl_aBCd4c8b2c,
    4,
    "4D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd8b16a2b,
    dnnl_ABcd8b16a2b,
    4,
    "4D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd8b16c2b,
    dnnl_aBCd8b16c2b,
    4,
    "4D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    BAcd8a16b2a,
    dnnl_BAcd8a16b2a,
    4,
    "4D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd8b8a,
    dnnl_ABcd8b8a,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd8b8c,
    dnnl_aBCd8b8c,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd8b4c,
    dnnl_aBCd8b4c,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd8c16b2c,
    dnnl_aBCd8c16b2c,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde8a16b2a,
    dnnl_ABcde8a16b2a,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aCBd8b16c2b,
    dnnl_aCBd8b16c2b,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd8c8b,
    dnnl_aBCd8c8b,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    Abcde16a,
    dnnl_Abcde16a,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    Abcde32a,
    dnnl_Abcde32a,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde16a16b,
    dnnl_ABcde16a16b,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    BAcde8a16b2a,
    dnnl_BAcde8a16b2a,
    4,
    "4D tensor blocked by 1st and 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd2b4c2b,
    dnnl_aBCd2b4c2b,
    4,
    "4D tensor blocked by 3rd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b16a4b,
    dnnl_ABcde4b16a4b,
    5,
    "5D tensor blocked by 1st dimension with block size 16"
);
impl_format_tag!(
    ABcde2b8a4b,
    dnnl_ABcde2b8a4b,
    5,
    "5D tensor blocked by 1st dimension with block size 8"
);
impl_format_tag!(
    aBcde16b,
    dnnl_aBcde16b,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    ABcde16b16a,
    dnnl_ABcde16b16a,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCde16b16c,
    dnnl_aBCde16b16c,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCde16c16b,
    dnnl_aBCde16c16b,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCde2c8b4c,
    dnnl_aBCde2c8b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    Abcde4a,
    dnnl_Abcde4a,
    5,
    "5D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBcde32b,
    dnnl_aBcde32b,
    5,
    "5D tensor blocked by 2nd dimension with block size 32"
);
impl_format_tag!(
    aBcde4b,
    dnnl_aBcde4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b4a,
    dnnl_ABcde4b4a,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4a4b,
    dnnl_ABcde4a4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde4b4c,
    dnnl_aBCde4b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde2c4b2c,
    dnnl_aBCde2c4b2c,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde4b8c2b,
    dnnl_aBCde4b8c2b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde4c16b4c,
    dnnl_aBCde4c16b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde16c16b4c,
    dnnl_aBCde16c16b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde16c16b2c,
    dnnl_aBCde16c16b2c,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde4c4b,
    dnnl_aBCde4c4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Abcde8a,
    dnnl_Abcde8a,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8a8b,
    dnnl_ABcde8a8b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8a4b,
    dnnl_ABcde8a4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcde16b16a,
    dnnl_BAcde16b16a,
    5,
    "5D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBcde8b,
    dnnl_aBcde8b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde8b16a2b,
    dnnl_ABcde8b16a2b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde8b16c2b,
    dnnl_aBCde8b16c2b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde4c8b2c,
    dnnl_aBCde4c8b2c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aCBde8b16c2b,
    dnnl_aCBde8b16c2b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde8b8a,
    dnnl_ABcde8b8a,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde32a32b,
    dnnl_ABcde32a32b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde8b8c,
    dnnl_aBCde8b8c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde8b4c,
    dnnl_aBCde8b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABc4a8b8a4b,
    dnnl_ABc4a8b8a4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd4a8b8a4b,
    dnnl_ABcd4a8b8a4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcde4a8b8a4b,
    dnnl_ABcde4a8b8a4b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    BAc4b8a8b4a,
    dnnl_BAc4b8a8b4a,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    BAcd4b8a8b4a,
    dnnl_BAcd4b8a8b4a,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    BAcde4b8a8b4a,
    dnnl_BAcde4b8a8b4a,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    ABcd2a8b8a2b,
    dnnl_ABcd2a8b8a2b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCd4b8c8b4c,
    dnnl_aBCd4b8c8b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde4b8c8b4c,
    dnnl_aBCde4b8c8b4c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde2b8c8b2c,
    dnnl_aBCde2b8c8b2c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde8c16b2c,
    dnnl_aBCde8c16b2c,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde8c8b,
    dnnl_aBCde8c8b,
    5,
    "5D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCde2b4c2b,
    dnnl_aBCde2b4c2b,
    5,
    "5D tensor blocked by 3rd dimension with block size 4"
);
impl_format_tag!(
    aBcdef16b,
    dnnl_aBcdef16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCdef16b16c,
    dnnl_aBCdef16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCdef16c16b,
    dnnl_aBCdef16c16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCdef4c16b4c,
    dnnl_aBCdef4c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 16"
);
impl_format_tag!(
    aBCdef2c8b4c,
    dnnl_aBCdef2c8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCdef4c8b2c,
    dnnl_aBCdef4c8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 8"
);
impl_format_tag!(
    aBCdef2b4c2b,
    dnnl_aBCdef2b4c2b,
    6,
    "6D tensor blocked by 3rd dimension with block size 4"
);
impl_format_tag!(
    aBcdef4b,
    dnnl_aBcdef4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef4c4b,
    dnnl_aBCdef4c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef4b4c,
    dnnl_aBCdef4b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef2c4b2c,
    dnnl_aBCdef2c4b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef4b8c2b,
    dnnl_aBCdef4b8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8b8c,
    dnnl_aBCdef8b8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8b4c,
    dnnl_aBCdef8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8c16b2c,
    dnnl_aBCdef8c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef4b8c8b4c,
    dnnl_aBCdef4b8c8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8b16c2b,
    dnnl_aBCdef8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef8b16c2b,
    dnnl_aCBdef8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8c8b,
    dnnl_aBCdef8c8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc16b,
    dnnl_aBdc16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16b2c,
    dnnl_aBdC16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16b4c,
    dnnl_aBdC16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc4b,
    dnnl_aBdc4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc8b,
    dnnl_aBdc8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec16b,
    dnnl_aBdec16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16b2c,
    dnnl_aBdeC16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16b4c,
    dnnl_aBdeC16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec32b,
    dnnl_aBdec32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec4b,
    dnnl_aBdec4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec8b,
    dnnl_aBdec8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc16b,
    dnnl_aBdefc16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16b2c,
    dnnl_aBdefC16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef16c16b,
    dnnl_aCBdef16c16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc4b,
    dnnl_aBdefc4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc8b,
    dnnl_aBdefc8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Abcdef16a,
    dnnl_Abcdef16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Abcdef32a,
    dnnl_Abcdef32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBedc16b,
    dnnl_aBedc16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb16a,
    dnnl_Acb16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16a2b,
    dnnl_AcB16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16a4b,
    dnnl_AcB16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb4a,
    dnnl_Acb4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb8a,
    dnnl_Acb8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBd16b16c,
    dnnl_aCBd16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBd16c16b,
    dnnl_aCBd16c16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBde16b16c,
    dnnl_aCBde16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBde16c16b,
    dnnl_aCBde16c16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb16a,
    dnnl_Acdb16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16a2b,
    dnnl_AcdB16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16a4b,
    dnnl_AcdB16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb32a,
    dnnl_Acdb32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb4a,
    dnnl_Acdb4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb8a,
    dnnl_Acdb8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb16a,
    dnnl_Acdeb16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16a2b,
    dnnl_AcdeB16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb4a,
    dnnl_Acdeb4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb8a,
    dnnl_Acdeb8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Adcb16a,
    dnnl_Adcb16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAc16a16b,
    dnnl_BAc16a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAc16b16a,
    dnnl_BAc16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcd16a16b,
    dnnl_BAcd16a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcd16b16a,
    dnnl_BAcd16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBd4c8b8c4b,
    dnnl_aCBd4c8b8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBde4c8b8c4b,
    dnnl_aCBde4c8b8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef4c8b8c4b,
    dnnl_aCBdef4c8b8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcde16a16b,
    dnnl_BAcde16a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef16b16c,
    dnnl_aCBdef16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b32a,
    dnnl_ABc16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b64a,
    dnnl_ABc16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b32a4b,
    dnnl_ABc4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b64a4b,
    dnnl_ABc4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b32a2b,
    dnnl_ABc8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b64a2b,
    dnnl_ABc8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b16a,
    dnnl_AB16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b32a,
    dnnl_AB16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b64a,
    dnnl_AB16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b16a2b,
    dnnl_AB8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b32a2b,
    dnnl_AB8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b64a2b,
    dnnl_AB8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB4b16a4b,
    dnnl_AB4b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB4b32a4b,
    dnnl_AB4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB4b64a4b,
    dnnl_AB4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b16a4b,
    dnnl_AB16b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b32a,
    dnnl_ABcd16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b64a,
    dnnl_ABcd16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b32a4b,
    dnnl_ABcd4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b64a4b,
    dnnl_ABcd4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b32a2b,
    dnnl_ABcd8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b64a2b,
    dnnl_ABcd8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b32a4b,
    dnnl_ABcde4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b64a4b,
    dnnl_ABcde4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b16a4b,
    dnnl_ABcde16b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b16a2b,
    dnnl_ABcde16b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b32a,
    dnnl_ABcde16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b64a,
    dnnl_ABcde16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b32a2b,
    dnnl_ABcde8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b64a2b,
    dnnl_ABcde8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef16c16b4c,
    dnnl_aBCdef16c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef16c16b2c,
    dnnl_aBCdef16c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB32a32b8a4b,
    dnnl_AB32a32b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8a4b,
    dnnl_AB8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB32a32b8a2b,
    dnnl_AB32a32b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8a2b,
    dnnl_AB8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abDc32d,
    dnnl_abDc32d,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abDC32d4c,
    dnnl_abDC32d4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEc32e,
    dnnl_abdEc32e,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEC32e2c,
    dnnl_abdEC32e2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEC32e4c,
    dnnl_abdEC32e4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16b4c,
    dnnl_aBdefC16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16a4b,
    dnnl_AcdeB16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16a16b2a,
    dnnl_ABcd16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16a16b2a,
    dnnl_ABc16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd16b16c2b,
    dnnl_aBCd16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde16b16c2b,
    dnnl_aBCde16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb32a,
    dnnl_Acb32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB32a2b,
    dnnl_AcB32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB32a4b,
    dnnl_AcB32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb48a,
    dnnl_Acb48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB48a2b,
    dnnl_AcB48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB48a4b,
    dnnl_AcB48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb64a,
    dnnl_Acb64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB64a2b,
    dnnl_AcB64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB64a4b,
    dnnl_AcB64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cBa2b,
    dnnl_cBa2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cBa4b,
    dnnl_cBa4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc32b,
    dnnl_aBdc32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC32b2c,
    dnnl_aBdC32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC32b4c,
    dnnl_aBdC32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc48b,
    dnnl_aBdc48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC48b2c,
    dnnl_aBdC48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC48b4c,
    dnnl_aBdC48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc64b,
    dnnl_aBdc64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC64b2c,
    dnnl_aBdC64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC64b4c,
    dnnl_aBdC64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adCb2c,
    dnnl_adCb2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adCb4c,
    dnnl_adCb4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB32a2b,
    dnnl_AcdB32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB32a4b,
    dnnl_AcdB32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb48a,
    dnnl_Acdb48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB48a2b,
    dnnl_AcdB48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB48a4b,
    dnnl_AcdB48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb64a,
    dnnl_Acdb64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB64a2b,
    dnnl_AcdB64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB64a4b,
    dnnl_AcdB64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cdBa2b,
    dnnl_cdBa2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cdBa4b,
    dnnl_cdBa4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC32b2c,
    dnnl_aBdeC32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC32b4c,
    dnnl_aBdeC32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec48b,
    dnnl_aBdec48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC48b2c,
    dnnl_aBdeC48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC48b4c,
    dnnl_aBdeC48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec64b,
    dnnl_aBdec64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC64b2c,
    dnnl_aBdeC64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC64b4c,
    dnnl_aBdeC64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adeCb2c,
    dnnl_adeCb2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adeCb4c,
    dnnl_adeCb4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb32a,
    dnnl_Acdeb32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB32a2b,
    dnnl_AcdeB32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB32a4b,
    dnnl_AcdeB32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb48a,
    dnnl_Acdeb48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB48a2b,
    dnnl_AcdeB48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB48a4b,
    dnnl_AcdeB48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb64a,
    dnnl_Acdeb64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB64a2b,
    dnnl_AcdeB64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB64a4b,
    dnnl_AcdeB64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cdeBa2b,
    dnnl_cdeBa2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cdeBa4b,
    dnnl_cdeBa4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc32b,
    dnnl_aBdefc32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC32b2c,
    dnnl_aBdefC32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC32b4c,
    dnnl_aBdefC32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc48b,
    dnnl_aBdefc48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC48b2c,
    dnnl_aBdefC48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC48b4c,
    dnnl_aBdefC48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc64b,
    dnnl_aBdefc64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC64b2c,
    dnnl_aBdefC64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC64b4c,
    dnnl_aBdefC64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adefCb2c,
    dnnl_adefCb2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    adefCb4c,
    dnnl_adefCb4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b32a4b,
    dnnl_AB16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b48a4b,
    dnnl_AB16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b64a4b,
    dnnl_AB16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b16a2b,
    dnnl_AB16b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b32a2b,
    dnnl_AB16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b48a2b,
    dnnl_AB16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b64a2b,
    dnnl_AB16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b32a4b,
    dnnl_ABc16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b48a4b,
    dnnl_ABc16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b64a4b,
    dnnl_ABc16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b32a2b,
    dnnl_ABc16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b48a2b,
    dnnl_ABc16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b64a2b,
    dnnl_ABc16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b32a4b,
    dnnl_ABcd16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b48a4b,
    dnnl_ABcd16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b64a4b,
    dnnl_ABcd16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b32a2b,
    dnnl_ABcd16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b48a2b,
    dnnl_ABcd16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b64a2b,
    dnnl_ABcd16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b32a4b,
    dnnl_ABcde16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b48a4b,
    dnnl_ABcde16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b64a4b,
    dnnl_ABcde16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b32a2b,
    dnnl_ABcde16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b48a2b,
    dnnl_ABcde16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b64a2b,
    dnnl_ABcde16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc32a16b,
    dnnl_ABc32a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd32a16b,
    dnnl_ABcd32a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde32a16b,
    dnnl_ABcde32a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB48a16b,
    dnnl_AB48a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB48a32b,
    dnnl_AB48a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc40a16b,
    dnnl_ABc40a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc40a32b,
    dnnl_ABc40a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC48b16c,
    dnnl_aBC48b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC48b32c,
    dnnl_aBC48b32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd40a16b,
    dnnl_ABcd40a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd40a32b,
    dnnl_ABcd40a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCd32c,
    dnnl_abCd32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdCe32c,
    dnnl_abdCe32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdCE32c2e,
    dnnl_abdCE32c2e,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a16b2a,
    dnnl_BA16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a32b2a,
    dnnl_BA16a32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a48b2a,
    dnnl_BA16a48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a64b2a,
    dnnl_BA16a64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a16b4a,
    dnnl_BA16a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a32b4a,
    dnnl_BA16a32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a48b4a,
    dnnl_BA16a48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a64b4a,
    dnnl_BA16a64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8a2b,
    dnnl_ABcd8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c16b2c,
    dnnl_aBdeC16c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c16b4c,
    dnnl_aBdeC16c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c16b2c,
    dnnl_aBdefC16c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b16a2b,
    dnnl_AcB16b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b16a4b,
    dnnl_AcB16b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b16a2b,
    dnnl_AcdB16b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b16a4b,
    dnnl_AcdB16b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b16a2b,
    dnnl_AcdeB16b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c16b4c,
    dnnl_aBdefC16c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b16a4b,
    dnnl_AcdeB16b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b32a2b,
    dnnl_AcB16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b32a4b,
    dnnl_AcB16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b48a2b,
    dnnl_AcB16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b48a4b,
    dnnl_AcB16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b64a2b,
    dnnl_AcB16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b64a4b,
    dnnl_AcB16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c16b2c,
    dnnl_aBdC16c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c16b4c,
    dnnl_aBdC16c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c32b2c,
    dnnl_aBdC16c32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c32b4c,
    dnnl_aBdC16c32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c48b2c,
    dnnl_aBdC16c48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c48b4c,
    dnnl_aBdC16c48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c64b2c,
    dnnl_aBdC16c64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC16c64b4c,
    dnnl_aBdC16c64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b32a2b,
    dnnl_AcdB16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b32a4b,
    dnnl_AcdB16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b48a2b,
    dnnl_AcdB16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b48a4b,
    dnnl_AcdB16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b64a2b,
    dnnl_AcdB16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b64a4b,
    dnnl_AcdB16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c32b2c,
    dnnl_aBdeC16c32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c32b4c,
    dnnl_aBdeC16c32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c48b2c,
    dnnl_aBdeC16c48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c48b4c,
    dnnl_aBdeC16c48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c64b2c,
    dnnl_aBdeC16c64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC16c64b4c,
    dnnl_aBdeC16c64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b32a2b,
    dnnl_AcdeB16b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b32a4b,
    dnnl_AcdeB16b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b48a2b,
    dnnl_AcdeB16b48a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b48a4b,
    dnnl_AcdeB16b48a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b64a2b,
    dnnl_AcdeB16b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b64a4b,
    dnnl_AcdeB16b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c32b2c,
    dnnl_aBdefC16c32b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c32b4c,
    dnnl_aBdefC16c32b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c48b2c,
    dnnl_aBdefC16c48b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c48b4c,
    dnnl_aBdefC16c48b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c64b2c,
    dnnl_aBdefC16c64b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC16c64b4c,
    dnnl_aBdefC16c64b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    decbA16a,
    dnnl_decbA16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4a2b,
    dnnl_ABc4a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8a2b,
    dnnl_ABc8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd8b2c,
    dnnl_aBCd8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4a2b,
    dnnl_ABcde4a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8a2b,
    dnnl_ABcde8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde40a16b,
    dnnl_ABcde40a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde40a32b,
    dnnl_ABcde40a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde8b2c,
    dnnl_aBCde8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4a8b8a2b,
    dnnl_ABcde4a8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4a8b8a2b,
    dnnl_ABcd4a8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4a8b8a2b,
    dnnl_ABc4a8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef4b8c8b2c,
    dnnl_aBCdef4b8c8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde4b8c8b2c,
    dnnl_aBCde4b8c8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd4b8c8b2c,
    dnnl_aBCd4b8c8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcde4b8a8b2a,
    dnnl_BAcde4b8a8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcd4b8a8b2a,
    dnnl_BAcd4b8a8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAc4b8a8b2a,
    dnnl_BAc4b8a8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef4c8b8c2b,
    dnnl_aCBdef4c8b8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBde4c8b8c2b,
    dnnl_aCBde4c8b8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBd4c8b8c2b,
    dnnl_aCBd4c8b8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef8b2c,
    dnnl_aBCdef8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB32a16b,
    dnnl_AB32a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB32a32b,
    dnnl_AB32a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA4b8a8b2a,
    dnnl_BA4b8a8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA4b8a8b4a,
    dnnl_BA4b8a8b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC32b16c,
    dnnl_aBC32b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC32b32c,
    dnnl_aBC32b32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB4c8b8c2b,
    dnnl_aCB4c8b8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB4c8b8c4b,
    dnnl_aCB4c8b8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4a2b,
    dnnl_ABcd4a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc2b8a16b4a,
    dnnl_ABc2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd2b8a16b4a,
    dnnl_ABcd2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde2b8a16b4a,
    dnnl_ABcde2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc2a8b16a4b,
    dnnl_ABc2a8b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc2a8b16a2b,
    dnnl_ABc2a8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc2b32a8b,
    dnnl_ABc2b32a8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd2a8b16a4b,
    dnnl_ABcd2a8b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd2a8b16a2b,
    dnnl_ABcd2a8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBd2c8b16c2b,
    dnnl_aCBd2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd2b32a8b,
    dnnl_ABcd2b32a8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd2c8b16c2b,
    dnnl_aBCd2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde2a8b16a4b,
    dnnl_ABcde2a8b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde2a8b16a2b,
    dnnl_ABcde2a8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBde2c8b16c2b,
    dnnl_aCBde2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde2b32a8b,
    dnnl_ABcde2b32a8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC2b8c16b2c,
    dnnl_aBC2b8c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd2b8c16b2c,
    dnnl_aBCd2b8c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde2b8c16b2c,
    dnnl_aBCde2b8c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef2b8c16b2c,
    dnnl_aBCdef2b8c16b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcde2b8a16b4a,
    dnnl_BAcde2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcd2b8a16b4a,
    dnnl_BAcd2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAc2b8a16b4a,
    dnnl_BAc2b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcde2b8a16b2a,
    dnnl_BAcde2b8a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAcd2b8a16b2a,
    dnnl_BAcd2b8a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BAc2b8a16b2a,
    dnnl_BAc2b8a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde2c8b16c2b,
    dnnl_aBCde2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef2c8b16c2b,
    dnnl_aBCdef2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCBdef2c8b16c2b,
    dnnl_aCBdef2c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCd2b8c16b4c,
    dnnl_aBCd2b8c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCde2b8c16b4c,
    dnnl_aBCde2b8c16b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA4b8a16b2a,
    dnnl_BA4b8a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA4b8a16b4a,
    dnnl_BA4b8a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB4c8b16c2b,
    dnnl_aCB4c8b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB4c8b16c4b,
    dnnl_aCB4c8b16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a16b,
    dnnl_BA16a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a32b,
    dnnl_BA16a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a48b,
    dnnl_BA16a48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16a64b,
    dnnl_BA16a64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16c2b,
    dnnl_aCB16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16c4b,
    dnnl_aCB16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16b2a,
    dnnl_BA16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA16b4a,
    dnnl_BA16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC16b16c,
    dnnl_aBC16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBC16b32c,
    dnnl_aBC16b32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16a16b,
    dnnl_AB16a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16a32b,
    dnnl_AB16a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16a16b2a,
    dnnl_ABcde16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBCdef16b16c2b,
    dnnl_aBCdef16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acedb16a,
    dnnl_Acedb16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdfec16b,
    dnnl_aBdfec16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEC64e2c,
    dnnl_abdEC64e2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEC64e4c,
    dnnl_abdEC64e4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b16c,
    dnnl_aCB16b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b32c,
    dnnl_aCB16b32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b48c,
    dnnl_aCB16b48c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b64c,
    dnnl_aCB16b64c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b16c2b,
    dnnl_aCB16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b32c2b,
    dnnl_aCB16b32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b48c2b,
    dnnl_aCB16b48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b64c2b,
    dnnl_aCB16b64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b16c4b,
    dnnl_aCB16b16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b32c4b,
    dnnl_aCB16b32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b48c4b,
    dnnl_aCB16b48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB16b64c4b,
    dnnl_aCB16b64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCd4c,
    dnnl_abCd4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCde4c,
    dnnl_abCde4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCdef4c,
    dnnl_abCdef4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCde32c,
    dnnl_abCde32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abCdef32c,
    dnnl_abCdef32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16a32b,
    dnnl_ABcd16a32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    decbA8a,
    dnnl_decbA8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b32c2b,
    dnnl_aCdefB16b32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b32c4b,
    dnnl_aCdefB16b32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b48c2b,
    dnnl_aCdefB16b48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b48c4b,
    dnnl_aCdefB16b48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b64c2b,
    dnnl_aCdefB16b64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b64c4b,
    dnnl_aCdefB16b64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a32b2a,
    dnnl_BcdeA16a32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a32b4a,
    dnnl_BcdeA16a32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a48b2a,
    dnnl_BcdeA16a48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a48b4a,
    dnnl_BcdeA16a48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a64b2a,
    dnnl_BcdeA16a64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a64b4a,
    dnnl_BcdeA16a64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb32c,
    dnnl_aCdefb32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB32c2b,
    dnnl_aCdefB32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB32c4b,
    dnnl_aCdefB32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb48c,
    dnnl_aCdefb48c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB48c2b,
    dnnl_aCdefB48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB48c4b,
    dnnl_aCdefB48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb64c,
    dnnl_aCdefb64c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB64c2b,
    dnnl_aCdefB64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB64c4b,
    dnnl_aCdefB64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea32b,
    dnnl_Bcdea32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA32b2a,
    dnnl_BcdeA32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA32b4a,
    dnnl_BcdeA32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea48b,
    dnnl_Bcdea48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA48b2a,
    dnnl_BcdeA48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA48b4a,
    dnnl_BcdeA48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea64b,
    dnnl_Bcdea64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA64b2a,
    dnnl_BcdeA64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA64b4a,
    dnnl_BcdeA64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca32b,
    dnnl_Bca32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA32b2a,
    dnnl_BcA32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA32b4a,
    dnnl_BcA32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca48b,
    dnnl_Bca48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA48b2a,
    dnnl_BcA48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA48b4a,
    dnnl_BcA48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca64b,
    dnnl_Bca64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA64b2a,
    dnnl_BcA64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA64b4a,
    dnnl_BcA64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb32c,
    dnnl_aCdb32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB32c2b,
    dnnl_aCdB32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB32c4b,
    dnnl_aCdB32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb48c,
    dnnl_aCdb48c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB48c2b,
    dnnl_aCdB48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB48c4b,
    dnnl_aCdB48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb64c,
    dnnl_aCdb64c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB64c2b,
    dnnl_aCdB64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB64c4b,
    dnnl_aCdB64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a16b2a,
    dnnl_BcA16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a16b4a,
    dnnl_BcA16a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a16b2a,
    dnnl_BcdA16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a16b4a,
    dnnl_BcdA16a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a16b2a,
    dnnl_BcdeA16a16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16a16b4a,
    dnnl_BcdeA16a16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b16c2b,
    dnnl_aCdB16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b16c4b,
    dnnl_aCdB16b16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b16c2b,
    dnnl_aCdeB16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b16c4b,
    dnnl_aCdeB16b16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b16c2b,
    dnnl_aCdefB16b16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16b16c4b,
    dnnl_aCdefB16b16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a32b2a,
    dnnl_BcA16a32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a32b4a,
    dnnl_BcA16a32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a48b2a,
    dnnl_BcA16a48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a48b4a,
    dnnl_BcA16a48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a64b2a,
    dnnl_BcA16a64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16a64b4a,
    dnnl_BcA16a64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b32c2b,
    dnnl_aCdB16b32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b32c4b,
    dnnl_aCdB16b32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b48c2b,
    dnnl_aCdB16b48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b48c4b,
    dnnl_aCdB16b48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b64c2b,
    dnnl_aCdB16b64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16b64c4b,
    dnnl_aCdB16b64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a32b2a,
    dnnl_BcdA16a32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a32b4a,
    dnnl_BcdA16a32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a48b2a,
    dnnl_BcdA16a48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a48b4a,
    dnnl_BcdA16a48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a64b2a,
    dnnl_BcdA16a64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16a64b4a,
    dnnl_BcdA16a64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b32c2b,
    dnnl_aCdeB16b32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b32c4b,
    dnnl_aCdeB16b32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b48c2b,
    dnnl_aCdeB16b48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b48c4b,
    dnnl_aCdeB16b48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b64c2b,
    dnnl_aCdeB16b64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16b64c4b,
    dnnl_aCdeB16b64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca16b,
    dnnl_Bca16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16b2a,
    dnnl_BcA16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA16b4a,
    dnnl_BcA16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda16b,
    dnnl_Bcda16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16b2a,
    dnnl_BcdA16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA16b4a,
    dnnl_BcdA16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea16b,
    dnnl_Bcdea16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16b2a,
    dnnl_BcdeA16b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA16b4a,
    dnnl_BcdeA16b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb16c,
    dnnl_aCdb16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16c2b,
    dnnl_aCdB16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB16c4b,
    dnnl_aCdB16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb16c,
    dnnl_aCdeb16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16c2b,
    dnnl_aCdeB16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB16c4b,
    dnnl_aCdeB16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb16c,
    dnnl_aCdefb16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16c2b,
    dnnl_aCdefB16c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB16c4b,
    dnnl_aCdefB16c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda32b,
    dnnl_Bcda32b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA32b2a,
    dnnl_BcdA32b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA32b4a,
    dnnl_BcdA32b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda48b,
    dnnl_Bcda48b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA48b2a,
    dnnl_BcdA48b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA48b4a,
    dnnl_BcdA48b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda64b,
    dnnl_Bcda64b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA64b2a,
    dnnl_BcdA64b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA64b4a,
    dnnl_BcdA64b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb32c,
    dnnl_aCdeb32c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB32c2b,
    dnnl_aCdeB32c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB32c4b,
    dnnl_aCdeB32c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb48c,
    dnnl_aCdeb48c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB48c2b,
    dnnl_aCdeB48c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB48c4b,
    dnnl_aCdeB48c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb64c,
    dnnl_aCdeb64c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB64c2b,
    dnnl_aCdeB64c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB64c4b,
    dnnl_aCdeB64c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acb24a,
    dnnl_Acb24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdb24a,
    dnnl_Acdb24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Acdeb24a,
    dnnl_Acdeb24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdc24b,
    dnnl_aBdc24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdec24b,
    dnnl_aBdec24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefc24b,
    dnnl_aBdefc24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abDc16d,
    dnnl_abDc16d,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdEc16e,
    dnnl_abdEc16e,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    abdCe16c,
    dnnl_abdCe16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB24a2b,
    dnnl_AcB24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB24a2b,
    dnnl_AcdB24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB24a2b,
    dnnl_AcdeB24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC24b2c,
    dnnl_aBdC24b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC24b2c,
    dnnl_aBdeC24b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC24b2c,
    dnnl_aBdefC24b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8a2b,
    dnnl_AcB8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8a2b,
    dnnl_AcdB8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8a2b,
    dnnl_AcdeB8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC8b2c,
    dnnl_aBdC8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC8b2c,
    dnnl_aBdeC8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC8b2c,
    dnnl_aBdefC8b2c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b32a,
    dnnl_AB8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b32a,
    dnnl_ABc8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b32a,
    dnnl_ABcd8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b32a,
    dnnl_ABcde8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b24a,
    dnnl_AB8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b24a,
    dnnl_ABc8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b24a,
    dnnl_ABcd8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b24a,
    dnnl_ABcde8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b16a,
    dnnl_AB8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b16a,
    dnnl_ABc8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b16a,
    dnnl_ABcd8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b16a,
    dnnl_ABcde8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b8a,
    dnnl_AB8b8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB4b8a4b,
    dnnl_AB4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB4b24a4b,
    dnnl_AB4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b8a4b,
    dnnl_ABc4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc4b24a4b,
    dnnl_ABc4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b8a4b,
    dnnl_ABcd4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd4b24a4b,
    dnnl_ABcd4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b8a4b,
    dnnl_ABcde4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde4b24a4b,
    dnnl_ABcde4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b24a2b,
    dnnl_AB8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b24a2b,
    dnnl_ABc8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b24a2b,
    dnnl_ABcd8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b24a2b,
    dnnl_ABcde8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB8b8a2b,
    dnnl_AB8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc8b8a2b,
    dnnl_ABc8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd8b8a2b,
    dnnl_ABcd8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde8b8a2b,
    dnnl_ABcde8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB24a4b,
    dnnl_AcB24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB24a4b,
    dnnl_AcdB24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB24a4b,
    dnnl_AcdeB24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC24b4c,
    dnnl_aBdC24b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC24b4c,
    dnnl_aBdeC24b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC24b4c,
    dnnl_aBdefC24b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8a4b,
    dnnl_AcB8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8a4b,
    dnnl_AcdB8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8a4b,
    dnnl_AcdeB8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdC8b4c,
    dnnl_aBdC8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdeC8b4c,
    dnnl_aBdeC8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aBdefC8b4c,
    dnnl_aBdefC8b4c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca8b,
    dnnl_Bca8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA8b2a,
    dnnl_BcA8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda8b,
    dnnl_Bcda8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA8b2a,
    dnnl_BcdA8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea8b,
    dnnl_Bcdea8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA8b2a,
    dnnl_BcdeA8b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb8c,
    dnnl_aCdb8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB8c2b,
    dnnl_aCdB8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb8c,
    dnnl_aCdeb8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB8c2b,
    dnnl_aCdeB8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb8c,
    dnnl_aCdefb8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB8c2b,
    dnnl_aCdefB8c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bca24b,
    dnnl_Bca24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA24b2a,
    dnnl_BcA24b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcda24b,
    dnnl_Bcda24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA24b2a,
    dnnl_BcdA24b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Bcdea24b,
    dnnl_Bcdea24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA24b2a,
    dnnl_BcdeA24b2a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdb24c,
    dnnl_aCdb24c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB24c2b,
    dnnl_aCdB24c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeb24c,
    dnnl_aCdeb24c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB24c2b,
    dnnl_aCdeB24c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefb24c,
    dnnl_aCdefb24c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB24c2b,
    dnnl_aCdefB24c2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA8b4a,
    dnnl_BcA8b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA8b4a,
    dnnl_BcdA8b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA8b4a,
    dnnl_BcdeA8b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB8c4b,
    dnnl_aCdB8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB8c4b,
    dnnl_aCdeB8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB8c4b,
    dnnl_aCdefB8c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcA24b4a,
    dnnl_BcA24b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdA24b4a,
    dnnl_BcdA24b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BcdeA24b4a,
    dnnl_BcdeA24b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdB24c4b,
    dnnl_aCdB24c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdeB24c4b,
    dnnl_aCdeB24c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCdefB24c4b,
    dnnl_aCdefB24c4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AB16b48a,
    dnnl_AB16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16b48a,
    dnnl_ABc16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16b48a,
    dnnl_ABcd16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16b48a,
    dnnl_ABcde16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABc16a4b,
    dnnl_ABc16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcd16a4b,
    dnnl_ABcd16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    ABcde16a4b,
    dnnl_ABcde16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    defcbA16a,
    dnnl_defcbA16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    defcbA8a,
    dnnl_defcbA8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b64a,
    dnnl_AcB16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b64a,
    dnnl_AcdB16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b64a,
    dnnl_AcdeB16b64a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b48a,
    dnnl_AcB16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b48a,
    dnnl_AcdB16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b48a,
    dnnl_AcdeB16b48a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b32a,
    dnnl_AcB16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b32a,
    dnnl_AcdB16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b32a,
    dnnl_AcdeB16b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB16b16a,
    dnnl_AcB16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB16b16a,
    dnnl_AcdB16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB16b16a,
    dnnl_AcdeB16b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b32a,
    dnnl_AcB8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b32a,
    dnnl_AcdB8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b32a,
    dnnl_AcdeB8b32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b24a,
    dnnl_AcB8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b24a,
    dnnl_AcdB8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b24a,
    dnnl_AcdeB8b24a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b16a,
    dnnl_AcB8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b16a,
    dnnl_AcdB8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b16a,
    dnnl_AcdeB8b16a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b8a,
    dnnl_AcB8b8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b8a,
    dnnl_AcdB8b8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b8a,
    dnnl_AcdeB8b8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b64a2b,
    dnnl_AcB8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b64a2b,
    dnnl_AcdB8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b64a2b,
    dnnl_AcdeB8b64a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b32a2b,
    dnnl_AcB8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b32a2b,
    dnnl_AcdB8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b32a2b,
    dnnl_AcdeB8b32a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b24a2b,
    dnnl_AcB8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b24a2b,
    dnnl_AcdB8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b24a2b,
    dnnl_AcdeB8b24a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b16a2b,
    dnnl_AcB8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b16a2b,
    dnnl_AcdB8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b16a2b,
    dnnl_AcdeB8b16a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB8b8a2b,
    dnnl_AcB8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB8b8a2b,
    dnnl_AcdB8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB8b8a2b,
    dnnl_AcdeB8b8a2b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB4b64a4b,
    dnnl_AcB4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB4b64a4b,
    dnnl_AcdB4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB4b64a4b,
    dnnl_AcdeB4b64a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB4b32a4b,
    dnnl_AcB4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB4b32a4b,
    dnnl_AcdB4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB4b32a4b,
    dnnl_AcdeB4b32a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB4b24a4b,
    dnnl_AcB4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB4b24a4b,
    dnnl_AcdB4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB4b24a4b,
    dnnl_AcdeB4b24a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB4b16a4b,
    dnnl_AcB4b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB4b16a4b,
    dnnl_AcdB4b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB4b16a4b,
    dnnl_AcdeB4b16a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcB4b8a4b,
    dnnl_AcB4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdB4b8a4b,
    dnnl_AcdB4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    AcdeB4b8a4b,
    dnnl_AcdeB4b8a4b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Ab4a,
    dnnl_Ab4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Ab8a,
    dnnl_Ab8a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA4b4a,
    dnnl_BA4b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA8b4a,
    dnnl_BA8b4a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA2a24b,
    dnnl_BA2a24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB2b24c,
    dnnl_aCB2b24c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA2a8b,
    dnnl_BA2a8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB2b8c,
    dnnl_aCB2b8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA8a24b,
    dnnl_BA8a24b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB8b24c,
    dnnl_aCB8b24c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA8a16b,
    dnnl_BA8a16b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB8b16c,
    dnnl_aCB8b16c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    BA8a8b,
    dnnl_BA8a8b,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    aCB8b8c,
    dnnl_aCB8b8c,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    bcad,
    dnnl_bcad,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    cabd,
    dnnl_cabd,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    dabc,
    dnnl_dabc,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(
    Ab32a,
    dnnl_Ab32a,
    6,
    "6D tensor blocked by 2nd dimension with block size 4"
);
impl_format_tag!(x, dnnl_x, 1, "1D tensor, an alias to #dnnl_a");
impl_format_tag!(
    nc,
    dnnl_nc,
    2,
    "2D CNN activations tensor, an alias to #dnnl_ab"
);
impl_format_tag!(
    cn,
    dnnl_cn,
    2,
    "2D CNN activations tensor, an alias to #dnnl_ba"
);
impl_format_tag!(
    tn,
    dnnl_tn,
    2,
    "2D RNN statistics tensor, an alias to #dnnl_ab"
);
impl_format_tag!(
    nt,
    dnnl_nt,
    2,
    "2D RNN statistics tensor, an alias to #dnnl_ba"
);
impl_format_tag!(
    ncw,
    dnnl_ncw,
    3,
    "3D CNN activations tensor, an alias to #dnnl_abc"
);
impl_format_tag!(
    nwc,
    dnnl_nwc,
    3,
    "3D CNN activations tensor, an alias to #dnnl_acb"
);
impl_format_tag!(
    nchw,
    dnnl_nchw,
    4,
    "4D CNN activations tensor, an alias to #dnnl_abcd"
);
impl_format_tag!(
    nhwc,
    dnnl_nhwc,
    4,
    "4D CNN activations tensor, an alias to #dnnl_acdb"
);
impl_format_tag!(
    chwn,
    dnnl_chwn,
    4,
    "4D CNN activations tensor, an alias to #dnnl_bcda"
);
impl_format_tag!(
    ncdhw,
    dnnl_ncdhw,
    5,
    "5D CNN activations tensor, an alias to #dnnl_abcde"
);
impl_format_tag!(
    ndhwc,
    dnnl_ndhwc,
    5,
    "5D CNN activations tensor, an alias to #dnnl_acdeb"
);
impl_format_tag!(
    oi,
    dnnl_oi,
    2,
    "2D CNN weights tensor, an alias to #dnnl_ab"
);
impl_format_tag!(
    io,
    dnnl_io,
    2,
    "2D CNN weights tensor, an alias to #dnnl_ba"
);
impl_format_tag!(
    oiw,
    dnnl_oiw,
    3,
    "3D CNN weights tensor, an alias to #dnnl_abc"
);
impl_format_tag!(
    owi,
    dnnl_owi,
    3,
    "3D CNN weights tensor, an alias to #dnnl_acb"
);
impl_format_tag!(
    wio,
    dnnl_wio,
    3,
    "3D CNN weights tensor, an alias to #dnnl_cba"
);
impl_format_tag!(
    woi,
    dnnl_woi,
    3,
    "3D CNN weights tensor, an alias to #dnnl_cab"
);
impl_format_tag!(
    iwo,
    dnnl_iwo,
    3,
    "3D CNN weights tensor, an alias to #dnnl_bca"
);
impl_format_tag!(
    oihw,
    dnnl_oihw,
    4,
    "4D CNN weights tensor, an alias to #dnnl_abcd"
);
impl_format_tag!(
    hwio,
    dnnl_hwio,
    4,
    "4D CNN weights tensor, an alias to #dnnl_cdba"
);
impl_format_tag!(
    hwoi,
    dnnl_hwoi,
    4,
    "4D CNN weights tensor, an alias to #dnnl_cdab"
);
impl_format_tag!(
    ohwi,
    dnnl_ohwi,
    4,
    "4D CNN weights tensor, an alias to #dnnl_acdb"
);
impl_format_tag!(
    ihwo,
    dnnl_ihwo,
    4,
    "4D CNN weights tensor, an alias to #dnnl_bcda"
);
impl_format_tag!(
    iohw,
    dnnl_iohw,
    4,
    "4D CNN weights tensor, an alias to #dnnl_bacd"
);
impl_format_tag!(
    oidhw,
    dnnl_oidhw,
    5,
    "5D CNN weights tensor, an alias to #dnnl_abcde"
);
impl_format_tag!(
    iodhw,
    dnnl_iodhw,
    5,
    "5D CNN weights tensor, an alias to #dnnl_bacde"
);
impl_format_tag!(
    dhwio,
    dnnl_dhwio,
    5,
    "5D CNN weights tensor, an alias to #dnnl_cdeba"
);
impl_format_tag!(
    dhwoi,
    dnnl_dhwoi,
    5,
    "5D CNN weights tensor, an alias to #dnnl_cdeab"
);
impl_format_tag!(
    odhwi,
    dnnl_odhwi,
    5,
    "5D CNN weights tensor, an alias to #dnnl_acdeb"
);
impl_format_tag!(
    idhwo,
    dnnl_idhwo,
    5,
    "5D CNN weights tensor, an alias to #dnnl_bcdea"
);
impl_format_tag!(
    goiw,
    dnnl_goiw,
    4,
    "4D CNN weights tensor (incl. groups), an alias to #dnnl_abcd"
);
impl_format_tag!(
    gowi,
    dnnl_gowi,
    4,
    "4D CNN weights tensor (incl. groups), an alias to #dnnl_abdc"
);
impl_format_tag!(
    wigo,
    dnnl_wigo,
    4,
    "4D CNN weights tensor (incl. groups), an alias to #dnnl_dcab"
);
impl_format_tag!(
    goihw,
    dnnl_goihw,
    5,
    "5D CNN weights tensor (incl. groups), an alias to #dnnl_abcde"
);
impl_format_tag!(
    gohwi,
    dnnl_gohwi,
    5,
    "5D CNN weights tensor (incl. groups), an alias to #dnnl_abdec"
);
impl_format_tag!(
    hwigo,
    dnnl_hwigo,
    5,
    "5D CNN weights tensor (incl. groups), an alias to #dnnl_decab"
);
impl_format_tag!(
    giohw,
    dnnl_giohw,
    5,
    "5D CNN weights tensor (incl. groups), an alias to #dnnl_acbde"
);
impl_format_tag!(
    goidhw,
    dnnl_goidhw,
    6,
    "6D CNN weights tensor (incl. groups), an alias to #dnnl_abcdef"
);
impl_format_tag!(
    godhwi,
    dnnl_godhwi,
    6,
    "6D CNN weights tensor (incl. groups), an alias to #dnnl_abdefc"
);
impl_format_tag!(
    giodhw,
    dnnl_giodhw,
    6,
    "6D CNN weights tensor (incl. groups), an alias to #dnnl_acbdef"
);
impl_format_tag!(
    dhwigo,
    dnnl_dhwigo,
    6,
    "6D CNN weights tensor (incl. groups), an alias to #dnnl_defcab"
);
impl_format_tag!(tnc, dnnl_tnc, 3, "3D RNN data tensor in the format (seq_length, batch, input channels),\\n an alias to #dnnl_abc.");
impl_format_tag!(ntc, dnnl_ntc, 3, "3D RNN data tensor in the format (batch, seq_length, input channels),\\n an alias to #dnnl_bac.");
impl_format_tag!(ldnc, dnnl_ldnc, 4, "4D RNN states tensor in the format (num_layers, num_directions,\\n batch, state channels), an alias to #dnnl_abcd.");
impl_format_tag!(ldigo, dnnl_ldigo, 5, "5D RNN weights tensor in the format (num_layers, num_directions,\\n input_channels, num_gates, output_channels), an alias to #dnnl_abcde.\\n\\n - For LSTM cells, the gates order is input, forget, candidate\\n and output gate.\\n - For GRU cells, the gates order is update, reset and output gate.");
impl_format_tag!(ldgoi, dnnl_ldgoi, 5, "5D RNN weights tensor in the format (num_layers, num_directions,\\n num_gates, output_channels, input_channels), an alias to #dnnl_abdec.\\n\\n - For LSTM cells, the gates order is input, forget, candidate\\n and output gate.\\n - For GRU cells, the gates order is update, reset and output gate.");
impl_format_tag!(ldio, dnnl_ldio, 4, "4D LSTM projection tensor in the format (num_layers, num_directions,\\n num_channels_in_hidden_state, num_channels_in_recurrent_projection),\\n an alias to #dnnl_abcd.");
impl_format_tag!(ldoi, dnnl_ldoi, 4, "4D LSTM projection tensor in the format (num_layers, num_directions,\\n num_channels_in_recurrent_projection, num_channels_in_hidden_state),\\n an alias to #dnnl_abdc.");
impl_format_tag!(ldgo, dnnl_ldgo, 4, "4D RNN bias tensor in the format (num_layers, num_directions,\\n num_gates, output_channels), an alias to #dnnl_abcd.\\n\\n - For LSTM cells, the gates order is input, forget, candidate\\n and output gate.\\n - For GRU cells, the gates order is update, reset and output gate.");
impl_format_tag!(ldOi16o, dnnl_ldOi16o, 5, "5D LSTM projection tensor");
impl_format_tag!(ldOi32o, dnnl_ldOi32o, 5, "5D LSTM projection tensor");
impl_format_tag!(ldOI32o4i, dnnl_ldOI32o4i, 5, "5D LSTM projection tensor");
impl_format_tag!(ldIo32i, dnnl_ldIo32i, 5, "5D LSTM projection tensor");
impl_format_tag!(ldgOi16o, dnnl_ldgOi16o, 6, "6D RNN weights tensor");
impl_format_tag!(ldgOi32o, dnnl_ldgOi32o, 6, "6D RNN weights tensor");
impl_format_tag!(ldgOI32o2i, dnnl_ldgOI32o2i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgOI32o4i, dnnl_ldgOI32o4i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgOI64o2i, dnnl_ldgOI64o2i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgOI64o4i, dnnl_ldgOI64o4i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgIo16i, dnnl_ldgIo16i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgIo32i, dnnl_ldgIo32i, 6, "6D RNN weights tensor");
impl_format_tag!(ldgIO32i2o, dnnl_ldgIO32i2o, 6, "6D RNN weights tensor");
impl_format_tag!(nCdhw32c, dnnl_nCdhw32c, 5, "5D CNN activations tensor blocked by channels with block size 32,\\n an alias to #dnnl_aBcde32b");
impl_format_tag!(nCdhw16c, dnnl_nCdhw16c, 5, "5D CNN activations tensor blocked by channels with block size 16,\\n an alias to #dnnl_aBcde16b");
impl_format_tag!(
    nCdhw4c,
    dnnl_nCdhw4c,
    5,
    "5D CNN activations tensor blocked by channels with block size 4,\\n an alias to #dnnl_aBcde4b"
);
impl_format_tag!(
    nCdhw8c,
    dnnl_nCdhw8c,
    5,
    "5D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBcde8b"
);
impl_format_tag!(nChw32c, dnnl_nChw32c, 4, "4D CNN activations tensor blocked by channels with block size 32,\\n an alias to #dnnl_aBcd32b");
impl_format_tag!(nChw16c, dnnl_nChw16c, 4, "4D CNN activations tensor blocked by channels with block size 16,\\n an alias to #dnnl_aBcd16b");
impl_format_tag!(
    nChw4c,
    dnnl_nChw4c,
    4,
    "4D CNN activations tensor blocked by channels with block size 4,\\n an alias to #dnnl_aBcd4b"
);
impl_format_tag!(
    nChw8c,
    dnnl_nChw8c,
    4,
    "4D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBcd8b"
);
impl_format_tag!(
    nCw32c,
    dnnl_nCw32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 32,\\n an alias to #dnnl_aBc32b"
);
impl_format_tag!(
    nCw16c,
    dnnl_nCw16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 16,\\n an alias to #dnnl_aBc16b"
);
impl_format_tag!(
    nCw4c,
    dnnl_nCw4c,
    3,
    "3D CNN activations tensor blocked by channels with block size 4,\\n an alias to #dnnl_aBc4b"
);
impl_format_tag!(
    nCw8c,
    dnnl_nCw8c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw16n16c,
    dnnl_NCw16n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw16n16c,
    dnnl_NCdhw16n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw16n16c,
    dnnl_NChw16n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw32n16c,
    dnnl_NCw32n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw32n16c,
    dnnl_NChw32n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw16n32c,
    dnnl_NChw16n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw32n16c,
    dnnl_NCdhw32n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw32n32c,
    dnnl_NCw32n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw32n32c,
    dnnl_NChw32n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw32n32c,
    dnnl_NCdhw32n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i16o,
    dnnl_OI16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i32o,
    dnnl_OI16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i48o,
    dnnl_OI16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i64o,
    dnnl_OI16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i8o2i,
    dnnl_OI8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i16o2i,
    dnnl_OI8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i24o2i,
    dnnl_OI8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i32o2i,
    dnnl_OI8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i64o2i,
    dnnl_OI8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI4i8o4i,
    dnnl_OI4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI4i16o4i,
    dnnl_OI4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI4i24o4i,
    dnnl_OI4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI4i32o4i,
    dnnl_OI4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI4i64o4i,
    dnnl_OI4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i16o4i,
    dnnl_OI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i32o,
    dnnl_OI8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i24o,
    dnnl_OI8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i16o,
    dnnl_OI8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI8i8o,
    dnnl_OI8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw16o16i,
    dnnl_IOw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw16i16o,
    dnnl_IOw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i16o,
    dnnl_OIw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i16o,
    dnnl_OwI16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i32o,
    dnnl_OIw16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i32o,
    dnnl_OwI16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i48o,
    dnnl_OIw16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i48o,
    dnnl_OwI16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i64o,
    dnnl_OIw16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i64o,
    dnnl_OwI16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16o16i,
    dnnl_OIw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oiw16o,
    dnnl_Oiw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i8o4i,
    dnnl_OIw4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI4i8o4i,
    dnnl_OwI4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i16o4i,
    dnnl_OIw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI4i16o4i,
    dnnl_OwI4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i24o4i,
    dnnl_OIw4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI4i24o4i,
    dnnl_OwI4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i32o4i,
    dnnl_OIw4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI4i32o4i,
    dnnl_OwI4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i64o4i,
    dnnl_OIw4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI4i64o4i,
    dnnl_OwI4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw2i8o4i,
    dnnl_OIw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i16o4i,
    dnnl_OIw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i16o2i,
    dnnl_OIw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16o16i2o,
    dnnl_OIw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4i4o,
    dnnl_OIw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4o4i,
    dnnl_OIw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oiw4o,
    dnnl_Oiw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i8o2i,
    dnnl_OIw8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i8o2i,
    dnnl_OwI8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i16o2i,
    dnnl_OIw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i16o2i,
    dnnl_OwI8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i24o2i,
    dnnl_OIw8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i24o2i,
    dnnl_OwI8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i32o2i,
    dnnl_OIw8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i32o2i,
    dnnl_OwI8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i64o2i,
    dnnl_OIw8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i64o2i,
    dnnl_OwI8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i8o,
    dnnl_OIw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i8o,
    dnnl_OwI8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8o16i2o,
    dnnl_OIw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw8o16i2o,
    dnnl_IOw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8o8i,
    dnnl_OIw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8o4i,
    dnnl_OIw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi16o,
    dnnl_Owi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16o2i,
    dnnl_OwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16o4i,
    dnnl_OwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo8i,
    dnnl_Iwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO8i2o,
    dnnl_IwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO8i4o,
    dnnl_IwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo16i,
    dnnl_Iwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16i2o,
    dnnl_IwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16i4o,
    dnnl_IwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo24i,
    dnnl_Iwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO24i2o,
    dnnl_IwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO24i4o,
    dnnl_IwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi4o,
    dnnl_Owi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi8o,
    dnnl_Owi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8o2i,
    dnnl_OwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i32o,
    dnnl_OIw8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i32o,
    dnnl_OwI8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i24o,
    dnnl_OIw8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i24o,
    dnnl_OwI8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw8i16o,
    dnnl_OIw8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8i16o,
    dnnl_OwI8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI8o4i,
    dnnl_OwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw16i16o,
    dnnl_IOhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw16o16i,
    dnnl_IOhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi16o,
    dnnl_Ohwi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16o2i,
    dnnl_OhwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16o4i,
    dnnl_OhwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo8i,
    dnnl_Ihwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO8i2o,
    dnnl_IhwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO8i4o,
    dnnl_IhwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo16i,
    dnnl_Ihwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16i2o,
    dnnl_IhwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16i4o,
    dnnl_IhwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo24i,
    dnnl_Ihwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO24i2o,
    dnnl_IhwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO24i4o,
    dnnl_IhwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi24o,
    dnnl_Ohwi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi32o,
    dnnl_Ohwi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi4o,
    dnnl_Ohwi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi8o,
    dnnl_Ohwi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8o2i,
    dnnl_OhwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8o4i,
    dnnl_OhwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i16o,
    dnnl_OIhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i16o,
    dnnl_OhwI16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i32o,
    dnnl_OIhw16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i32o,
    dnnl_OhwI16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i48o,
    dnnl_OIhw16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i48o,
    dnnl_OhwI16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i64o,
    dnnl_OIhw16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i64o,
    dnnl_OhwI16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16o16i,
    dnnl_OIhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oihw16o,
    dnnl_Oihw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i8o4i,
    dnnl_OIhw4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI4i8o4i,
    dnnl_OhwI4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i16o4i,
    dnnl_OIhw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI4i16o4i,
    dnnl_OhwI4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i24o4i,
    dnnl_OIhw4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI4i24o4i,
    dnnl_OhwI4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i32o4i,
    dnnl_OIhw4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI4i32o4i,
    dnnl_OhwI4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i64o4i,
    dnnl_OIhw4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI4i64o4i,
    dnnl_OhwI4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i16o4i,
    dnnl_OIhw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i16o2i,
    dnnl_OIhw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16o16i2o,
    dnnl_OIhw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4i4o,
    dnnl_OIhw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4o4i,
    dnnl_OIhw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oihw4o,
    dnnl_Oihw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i8o2i,
    dnnl_OIhw8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i8o2i,
    dnnl_OhwI8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i16o2i,
    dnnl_OIhw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i16o2i,
    dnnl_OhwI8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i32o2i,
    dnnl_OIhw8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i32o2i,
    dnnl_OhwI8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i24o2i,
    dnnl_OIhw8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i24o2i,
    dnnl_OhwI8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i64o2i,
    dnnl_OIhw8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i64o2i,
    dnnl_OhwI8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i8o,
    dnnl_OIhw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i8o,
    dnnl_OhwI8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8o16i2o,
    dnnl_OIhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw2i8o4i,
    dnnl_OIhw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw8o16i2o,
    dnnl_IOhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8o8i,
    dnnl_OIhw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8o4i,
    dnnl_OIhw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owhi16o,
    dnnl_Owhi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i32o,
    dnnl_OIhw8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i32o,
    dnnl_OhwI8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i24o,
    dnnl_OIhw8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i24o,
    dnnl_OhwI8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw8i16o,
    dnnl_OIhw8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI8i16o,
    dnnl_OhwI8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi16o,
    dnnl_Odhwi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16o2i,
    dnnl_OdhwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16o4i,
    dnnl_OdhwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo8i,
    dnnl_Idhwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO8i2o,
    dnnl_IdhwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO8i4o,
    dnnl_IdhwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo16i,
    dnnl_Idhwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16i2o,
    dnnl_IdhwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16i4o,
    dnnl_IdhwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo24i,
    dnnl_Idhwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO24i2o,
    dnnl_IdhwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO24i4o,
    dnnl_IdhwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi4o,
    dnnl_Odhwi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi8o,
    dnnl_Odhwi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8o2i,
    dnnl_OdhwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8o4i,
    dnnl_OdhwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odwhi16o,
    dnnl_Odwhi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i16o,
    dnnl_OIdhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i16o,
    dnnl_OdhwI16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i32o,
    dnnl_OIdhw16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i32o,
    dnnl_OdhwI16i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i48o,
    dnnl_OIdhw16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i48o,
    dnnl_OdhwI16i48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i64o,
    dnnl_OIdhw16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i64o,
    dnnl_OdhwI16i64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16o16i,
    dnnl_OIdhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oidhw16o,
    dnnl_Oidhw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i4o,
    dnnl_OIdhw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4o4i,
    dnnl_OIdhw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Oidhw4o,
    dnnl_Oidhw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i8o2i,
    dnnl_OIdhw8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i8o2i,
    dnnl_OdhwI8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i16o2i,
    dnnl_OIdhw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i16o2i,
    dnnl_OdhwI8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i32o2i,
    dnnl_OIdhw8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i32o2i,
    dnnl_OdhwI8i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i24o2i,
    dnnl_OIdhw8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i24o2i,
    dnnl_OdhwI8i24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i64o2i,
    dnnl_OIdhw8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i64o2i,
    dnnl_OdhwI8i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i8o,
    dnnl_OIdhw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i8o,
    dnnl_OdhwI8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8o16i2o,
    dnnl_OIdhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw8o16i2o,
    dnnl_IOdhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i8o4i,
    dnnl_OIdhw4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI4i8o4i,
    dnnl_OdhwI4i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i16o4i,
    dnnl_OIdhw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI4i16o4i,
    dnnl_OdhwI4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i24o4i,
    dnnl_OIdhw4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI4i24o4i,
    dnnl_OdhwI4i24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i32o4i,
    dnnl_OIdhw4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI4i32o4i,
    dnnl_OdhwI4i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4i64o4i,
    dnnl_OIdhw4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI4i64o4i,
    dnnl_OdhwI4i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i16o4i,
    dnnl_OIdhw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i16o2i,
    dnnl_OIdhw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw2i8o4i,
    dnnl_OIdhw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8o8i,
    dnnl_OIdhw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8o4i,
    dnnl_OIdhw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw16i16o,
    dnnl_IOdhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4o8i8o4i,
    dnnl_OIdhw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw16o16i,
    dnnl_IOdhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16o16i2o,
    dnnl_OIdhw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i32o,
    dnnl_OIdhw8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i32o,
    dnnl_OdhwI8i32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i24o,
    dnnl_OIdhw8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i24o,
    dnnl_OdhwI8i24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw8i16o,
    dnnl_OIdhw8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI8i16o,
    dnnl_OdhwI8i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goiw16g,
    dnnl_Goiw16g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goiw8g,
    dnnl_Goiw8g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goiw4g,
    dnnl_Goiw4g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw16o16i,
    dnnl_gIOw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw16i16o,
    dnnl_gIOw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw16i16o,
    dnnl_gOIw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw16o16i,
    dnnl_gOIw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOiw16o,
    dnnl_gOiw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4i16o4i,
    dnnl_gOIw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw2i8o4i,
    dnnl_gOIw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw16i16o4i,
    dnnl_gOIw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw16i16o2i,
    dnnl_gOIw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw16o16i2o,
    dnnl_gOIw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4i4o,
    dnnl_gOIw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4o4i,
    dnnl_gOIw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOiw4o,
    dnnl_gOiw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw8i16o2i,
    dnnl_gOIw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw8i8o,
    dnnl_gOIw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw8o16i2o,
    dnnl_gOIw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw8o16i2o,
    dnnl_gIOw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw8o8i,
    dnnl_gOIw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw8o4i,
    dnnl_gOIw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi16o,
    dnnl_gOwi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16o2i,
    dnnl_gOwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16o4i,
    dnnl_gOwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo8i,
    dnnl_gIwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO8i2o,
    dnnl_gIwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO8i4o,
    dnnl_gIwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo16i,
    dnnl_gIwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16i2o,
    dnnl_gIwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16i4o,
    dnnl_gIwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo24i,
    dnnl_gIwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO24i2o,
    dnnl_gIwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO24i4o,
    dnnl_gIwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi4o,
    dnnl_gOwi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi8o,
    dnnl_gOwi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI8o2i,
    dnnl_gOwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI8o4i,
    dnnl_gOwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goiw32g,
    dnnl_Goiw32g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw2i4o2i,
    dnnl_gOIw2i4o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw2o4i2o,
    dnnl_gOIw2o4i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4i8o2i,
    dnnl_gOIw4i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4o8i2o,
    dnnl_gOIw4o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIw4i,
    dnnl_goIw4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIw32i,
    dnnl_goIw32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw16i16o,
    dnnl_gIOhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw16o16i,
    dnnl_gIOhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi16o,
    dnnl_gOhwi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16o2i,
    dnnl_gOhwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16o4i,
    dnnl_gOhwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo8i,
    dnnl_gIhwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO8i2o,
    dnnl_gIhwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO8i4o,
    dnnl_gIhwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo16i,
    dnnl_gIhwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16i2o,
    dnnl_gIhwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16i4o,
    dnnl_gIhwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo24i,
    dnnl_gIhwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO24i2o,
    dnnl_gIhwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO24i4o,
    dnnl_gIhwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi32o,
    dnnl_gOhwi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi24o,
    dnnl_gOhwi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI24o2i,
    dnnl_gOhwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI24o4i,
    dnnl_gOhwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi4o,
    dnnl_gOhwi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi8o,
    dnnl_gOhwi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI8o2i,
    dnnl_gOhwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI8o4i,
    dnnl_gOhwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goihw16g,
    dnnl_Goihw16g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw16i16o,
    dnnl_gOIhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw16o16i,
    dnnl_gOIhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOihw16o,
    dnnl_gOihw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2i8o4i,
    dnnl_gOIhw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4i16o4i,
    dnnl_gOIhw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw16i16o4i,
    dnnl_gOIhw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw16i16o2i,
    dnnl_gOIhw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw16o16i2o,
    dnnl_gOIhw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4i4o,
    dnnl_gOIhw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4o4i,
    dnnl_gOIhw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOihw4o,
    dnnl_gOihw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goihw8g,
    dnnl_Goihw8g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goihw4g,
    dnnl_Goihw4g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw8i16o2i,
    dnnl_gOIhw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw8i8o,
    dnnl_gOIhw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw8o16i2o,
    dnnl_gOIhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw8o16i2o,
    dnnl_gIOhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw8o8i,
    dnnl_gOIhw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw8o4i,
    dnnl_gOIhw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goihw32g,
    dnnl_Goihw32g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwhi16o,
    dnnl_gOwhi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIhw4i,
    dnnl_goIhw4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIhw32i,
    dnnl_goIhw32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4o8i8o4i,
    dnnl_OIw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4o8i8o4i,
    dnnl_OIhw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw4i8o8i4o,
    dnnl_IOw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw4i8o8i4o,
    dnnl_IOhw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw4i8o8i4o,
    dnnl_IOdhw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw2o8i8o2i,
    dnnl_OIhw2o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4o8i8o4i,
    dnnl_gOIw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4o8i8o4i,
    dnnl_gOIhw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4o8i8o4i,
    dnnl_gOIdhw4o8i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw4i8o8i4o,
    dnnl_gIOw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw4i8o8i4o,
    dnnl_gIOhw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw4i8o8i4o,
    dnnl_gIOdhw4i8o8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2o8i8o2i,
    dnnl_gOIhw2o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2i4o2i,
    dnnl_gOIhw2i4o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2o4i2o,
    dnnl_gOIhw2o4i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4i8o2i,
    dnnl_gOIhw4i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4o8i2o,
    dnnl_gOIhw4o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw16i16o,
    dnnl_gIOdhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw16o16i,
    dnnl_gIOdhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi16o,
    dnnl_gOdhwi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16o2i,
    dnnl_gOdhwI16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16o4i,
    dnnl_gOdhwI16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo8i,
    dnnl_gIdhwo8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO8i2o,
    dnnl_gIdhwO8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO8i4o,
    dnnl_gIdhwO8i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo16i,
    dnnl_gIdhwo16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16i2o,
    dnnl_gIdhwO16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16i4o,
    dnnl_gIdhwO16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo24i,
    dnnl_gIdhwo24i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO24i2o,
    dnnl_gIdhwO24i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO24i4o,
    dnnl_gIdhwO24i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi4o,
    dnnl_gOdhwi4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi8o,
    dnnl_gOdhwi8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI8o2i,
    dnnl_gOdhwI8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI8o4i,
    dnnl_gOdhwI8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdwhi16o,
    dnnl_gOdwhi16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw16i16o,
    dnnl_gOIdhw16i16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4i16o4i,
    dnnl_gOIdhw4i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw16i16o4i,
    dnnl_gOIdhw16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw2i8o4i,
    dnnl_gOIdhw2i8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw16i16o2i,
    dnnl_gOIdhw16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw16o16i,
    dnnl_gOIdhw16o16i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw16o16i2o,
    dnnl_gOIdhw16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOidhw16o,
    dnnl_gOidhw16o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4i4o,
    dnnl_gOIdhw4i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4o4i,
    dnnl_gOIdhw4o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOidhw4o,
    dnnl_gOidhw4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw8i16o2i,
    dnnl_gOIdhw8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw8i8o,
    dnnl_gOIdhw8i8o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw8o16i2o,
    dnnl_gOIdhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw8o16i2o,
    dnnl_gIOdhw8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw8o8i,
    dnnl_gOIdhw8o8i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw8o4i,
    dnnl_gOIdhw8o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goidhw16g,
    dnnl_Goidhw16g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Goidhw32g,
    dnnl_Goidhw32g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw2i4o2i,
    dnnl_gOIdhw2i4o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4i8o2i,
    dnnl_gOIdhw4i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw2o4i2o,
    dnnl_gOIdhw2o4i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4o8i2o,
    dnnl_gOIdhw4o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIdhw4i,
    dnnl_goIdhw4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    goIdhw32i,
    dnnl_goIdhw32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi24o,
    dnnl_Owi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI24o2i,
    dnnl_OwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI24o4i,
    dnnl_OwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi32o,
    dnnl_Owi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI32o2i,
    dnnl_OwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI32o4i,
    dnnl_OwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi48o,
    dnnl_Owi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI48o2i,
    dnnl_OwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI48o4i,
    dnnl_OwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Owi64o,
    dnnl_Owi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI64o2i,
    dnnl_OwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI64o4i,
    dnnl_OwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo32i,
    dnnl_Iwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO32i2o,
    dnnl_IwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO32i4o,
    dnnl_IwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo48i,
    dnnl_Iwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO48i2o,
    dnnl_IwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO48i4o,
    dnnl_IwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Iwo64i,
    dnnl_Iwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO64i2o,
    dnnl_IwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO64i4o,
    dnnl_IwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    wIo2i,
    dnnl_wIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    wIo4i,
    dnnl_wIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi24o,
    dnnl_gOwi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI24o2i,
    dnnl_gOwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI24o4i,
    dnnl_gOwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi32o,
    dnnl_gOwi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI32o2i,
    dnnl_gOwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI32o4i,
    dnnl_gOwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi48o,
    dnnl_gOwi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI48o2i,
    dnnl_gOwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI48o4i,
    dnnl_gOwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwi64o,
    dnnl_gOwi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI64o2i,
    dnnl_gOwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI64o4i,
    dnnl_gOwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo32i,
    dnnl_gIwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO32i2o,
    dnnl_gIwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO32i4o,
    dnnl_gIwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo48i,
    dnnl_gIwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO48i2o,
    dnnl_gIwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO48i4o,
    dnnl_gIwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwo64i,
    dnnl_gIwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO64i2o,
    dnnl_gIwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO64i4o,
    dnnl_gIwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gwio,
    dnnl_gwio,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gwIo2i,
    dnnl_gwIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gwIo4i,
    dnnl_gwIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI24o,
    dnnl_OhwI24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI24o2i,
    dnnl_OhwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI24o4i,
    dnnl_OhwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI32o,
    dnnl_OhwI32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI32o2i,
    dnnl_OhwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI32o4i,
    dnnl_OhwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi48o,
    dnnl_Ohwi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI48o2i,
    dnnl_OhwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI48o4i,
    dnnl_OhwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ohwi64o,
    dnnl_Ohwi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI64o2i,
    dnnl_OhwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI64o4i,
    dnnl_OhwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo32i,
    dnnl_Ihwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO32i2o,
    dnnl_IhwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO32i4o,
    dnnl_IhwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo48i,
    dnnl_Ihwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO48i2o,
    dnnl_IhwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO48i4o,
    dnnl_IhwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Ihwo64i,
    dnnl_Ihwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO64i2o,
    dnnl_IhwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO64i4o,
    dnnl_IhwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    hwIo2i,
    dnnl_hwIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    hwIo4i,
    dnnl_hwIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI24o,
    dnnl_gOhwI24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI32o,
    dnnl_gOhwI32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI32o2i,
    dnnl_gOhwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI32o4i,
    dnnl_gOhwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi48o,
    dnnl_gOhwi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI48o2i,
    dnnl_gOhwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI48o4i,
    dnnl_gOhwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwi64o,
    dnnl_gOhwi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI64o2i,
    dnnl_gOhwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI64o4i,
    dnnl_gOhwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo32i,
    dnnl_gIhwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO32i2o,
    dnnl_gIhwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO32i4o,
    dnnl_gIhwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo48i,
    dnnl_gIhwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO48i2o,
    dnnl_gIhwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO48i4o,
    dnnl_gIhwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwo64i,
    dnnl_gIhwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO64i2o,
    dnnl_gIhwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO64i4o,
    dnnl_gIhwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    ghwio,
    dnnl_ghwio,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    ghwIo2i,
    dnnl_ghwIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    ghwIo4i,
    dnnl_ghwIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi24o,
    dnnl_Odhwi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI24o2i,
    dnnl_OdhwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI24o4i,
    dnnl_OdhwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi32o,
    dnnl_Odhwi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI32o2i,
    dnnl_OdhwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI32o4i,
    dnnl_OdhwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi48o,
    dnnl_Odhwi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI48o2i,
    dnnl_OdhwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI48o4i,
    dnnl_OdhwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Odhwi64o,
    dnnl_Odhwi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI64o2i,
    dnnl_OdhwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI64o4i,
    dnnl_OdhwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo32i,
    dnnl_Idhwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO32i2o,
    dnnl_IdhwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO32i4o,
    dnnl_IdhwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo48i,
    dnnl_Idhwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO48i2o,
    dnnl_IdhwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO48i4o,
    dnnl_IdhwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    Idhwo64i,
    dnnl_Idhwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO64i2o,
    dnnl_IdhwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO64i4o,
    dnnl_IdhwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    dhwIo2i,
    dnnl_dhwIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    dhwIo4i,
    dnnl_dhwIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi24o,
    dnnl_gOdhwi24o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI24o2i,
    dnnl_gOdhwI24o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI24o4i,
    dnnl_gOdhwI24o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi32o,
    dnnl_gOdhwi32o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI32o2i,
    dnnl_gOdhwI32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI32o4i,
    dnnl_gOdhwI32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi48o,
    dnnl_gOdhwi48o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI48o2i,
    dnnl_gOdhwI48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI48o4i,
    dnnl_gOdhwI48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwi64o,
    dnnl_gOdhwi64o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI64o2i,
    dnnl_gOdhwI64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI64o4i,
    dnnl_gOdhwI64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo32i,
    dnnl_gIdhwo32i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO32i2o,
    dnnl_gIdhwO32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO32i4o,
    dnnl_gIdhwO32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo48i,
    dnnl_gIdhwo48i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO48i2o,
    dnnl_gIdhwO48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO48i4o,
    dnnl_gIdhwO48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwo64i,
    dnnl_gIdhwo64i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO64i2o,
    dnnl_gIdhwO64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO64i4o,
    dnnl_gIdhwO64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gdhwio,
    dnnl_gdhwio,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gdhwIo2i,
    dnnl_gdhwIo2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gdhwIo4i,
    dnnl_gdhwIo4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i32o4i,
    dnnl_OI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i48o4i,
    dnnl_OI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i64o4i,
    dnnl_OI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i16o2i,
    dnnl_OI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i32o2i,
    dnnl_OI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i48o2i,
    dnnl_OI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OI16i64o2i,
    dnnl_OI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i32o4i,
    dnnl_OIw16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i48o4i,
    dnnl_OIw16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i64o4i,
    dnnl_OIw16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i32o2i,
    dnnl_OIw16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i48o2i,
    dnnl_OIw16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw16i64o2i,
    dnnl_OIw16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i32o4i,
    dnnl_OIhw16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i48o4i,
    dnnl_OIhw16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i64o4i,
    dnnl_OIhw16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i32o2i,
    dnnl_OIhw16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i48o2i,
    dnnl_OIhw16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw16i64o2i,
    dnnl_OIhw16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i32o4i,
    dnnl_OIdhw16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i48o4i,
    dnnl_OIdhw16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i64o4i,
    dnnl_OIdhw16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i32o2i,
    dnnl_OIdhw16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i48o2i,
    dnnl_OIdhw16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw16i64o2i,
    dnnl_OIdhw16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i16o2i,
    dnnl_OwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i16o4i,
    dnnl_OwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i16o2i,
    dnnl_OhwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i16o4i,
    dnnl_OhwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i16o2i,
    dnnl_OdhwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i16o4i,
    dnnl_OdhwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o16i2o,
    dnnl_IwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o16i4o,
    dnnl_IwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o16i2o,
    dnnl_IhwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o16i4o,
    dnnl_IhwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o16i2o,
    dnnl_IdhwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o16i4o,
    dnnl_IdhwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i16o2i,
    dnnl_gOwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i16o4i,
    dnnl_gOwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i16o2i,
    dnnl_gOhwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i16o4i,
    dnnl_gOhwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i16o2i,
    dnnl_gOdhwI16i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i16o4i,
    dnnl_gOdhwI16i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o16i2o,
    dnnl_gIwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o16i4o,
    dnnl_gIwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o16i2o,
    dnnl_gIhwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o16i4o,
    dnnl_gIhwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o16i2o,
    dnnl_gIdhwO16o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o16i4o,
    dnnl_gIdhwO16o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i32o2i,
    dnnl_OwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i32o4i,
    dnnl_OwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i48o2i,
    dnnl_OwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i48o4i,
    dnnl_OwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i64o2i,
    dnnl_OwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OwI16i64o4i,
    dnnl_OwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o32i2o,
    dnnl_IwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o32i4o,
    dnnl_IwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o48i2o,
    dnnl_IwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o48i4o,
    dnnl_IwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o64i2o,
    dnnl_IwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IwO16o64i4o,
    dnnl_IwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i32o2i,
    dnnl_gOwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i32o4i,
    dnnl_gOwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i48o2i,
    dnnl_gOwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i48o4i,
    dnnl_gOwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i64o2i,
    dnnl_gOwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOwI16i64o4i,
    dnnl_gOwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o32i2o,
    dnnl_gIwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o32i4o,
    dnnl_gIwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o48i2o,
    dnnl_gIwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o48i4o,
    dnnl_gIwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o64i2o,
    dnnl_gIwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIwO16o64i4o,
    dnnl_gIwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i32o2i,
    dnnl_OhwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i32o4i,
    dnnl_OhwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i48o2i,
    dnnl_OhwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i48o4i,
    dnnl_OhwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i64o2i,
    dnnl_OhwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OhwI16i64o4i,
    dnnl_OhwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o32i2o,
    dnnl_IhwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o32i4o,
    dnnl_IhwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o48i2o,
    dnnl_IhwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o48i4o,
    dnnl_IhwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o64i2o,
    dnnl_IhwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IhwO16o64i4o,
    dnnl_IhwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i32o2i,
    dnnl_gOhwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i32o4i,
    dnnl_gOhwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i48o2i,
    dnnl_gOhwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i48o4i,
    dnnl_gOhwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i64o2i,
    dnnl_gOhwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOhwI16i64o4i,
    dnnl_gOhwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o32i2o,
    dnnl_gIhwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o32i4o,
    dnnl_gIhwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o48i2o,
    dnnl_gIhwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o48i4o,
    dnnl_gIhwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o64i2o,
    dnnl_gIhwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIhwO16o64i4o,
    dnnl_gIhwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i32o2i,
    dnnl_OdhwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i32o4i,
    dnnl_OdhwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i48o2i,
    dnnl_OdhwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i48o4i,
    dnnl_OdhwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i64o2i,
    dnnl_OdhwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OdhwI16i64o4i,
    dnnl_OdhwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o32i2o,
    dnnl_IdhwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o32i4o,
    dnnl_IdhwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o48i2o,
    dnnl_IdhwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o48i4o,
    dnnl_IdhwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o64i2o,
    dnnl_IdhwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IdhwO16o64i4o,
    dnnl_IdhwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i32o2i,
    dnnl_gOdhwI16i32o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i32o4i,
    dnnl_gOdhwI16i32o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i48o2i,
    dnnl_gOdhwI16i48o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i48o4i,
    dnnl_gOdhwI16i48o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i64o2i,
    dnnl_gOdhwI16i64o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOdhwI16i64o4i,
    dnnl_gOdhwI16i64o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o32i2o,
    dnnl_gIdhwO16o32i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o32i4o,
    dnnl_gIdhwO16o32i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o48i2o,
    dnnl_gIdhwO16o48i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o48i4o,
    dnnl_gIdhwO16o48i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o64i2o,
    dnnl_gIdhwO16o64i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIdhwO16o64i4o,
    dnnl_gIdhwO16o64i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    hwioG16g,
    dnnl_hwioG16g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    hwioG8g,
    dnnl_hwioG8g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    dhwioG16g,
    dnnl_dhwioG16g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    dhwioG8g,
    dnnl_dhwioG8g,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw40n16c,
    dnnl_NCdhw40n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw40n16c,
    dnnl_NCw40n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw40n16c,
    dnnl_NChw40n16c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw40n32c,
    dnnl_NCw40n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw40n32c,
    dnnl_NChw40n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw40n32c,
    dnnl_NCdhw40n32c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw4o8i8o2i,
    dnnl_OIdhw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw4o8i8o2i,
    dnnl_OIhw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw4o8i8o2i,
    dnnl_OIw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw4o8i8o2i,
    dnnl_gOIdhw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw4o8i8o2i,
    dnnl_gOIhw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw4o8i8o2i,
    dnnl_gOIw4o8i8o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw4i8o8i2o,
    dnnl_IOdhw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw4i8o8i2o,
    dnnl_IOhw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw4i8o8i2o,
    dnnl_IOw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw4i8o8i2o,
    dnnl_gIOdhw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw4i8o8i2o,
    dnnl_gIOhw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw4i8o8i2o,
    dnnl_gIOw4i8o8i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCw2c32n8c,
    dnnl_NCw2c32n8c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NChw2c32n8c,
    dnnl_NChw2c32n8c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    NCdhw2c32n8c,
    dnnl_NCdhw2c32n8c,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw2i8o16i4o,
    dnnl_OIw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw2i8o16i4o,
    dnnl_OIhw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw2i8o16i4o,
    dnnl_OIdhw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw2o8i16o4i,
    dnnl_OIw2o8i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIw2o8i16o2i,
    dnnl_OIw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw2i8o16i4o,
    dnnl_IOw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOw2i8o16i2o,
    dnnl_IOw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw2o8i16o4i,
    dnnl_OIhw2o8i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIhw2o8i16o2i,
    dnnl_OIhw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw2i8o16i4o,
    dnnl_IOhw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOhw2i8o16i2o,
    dnnl_IOhw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw2o8i16o4i,
    dnnl_OIdhw2o8i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    OIdhw2o8i16o2i,
    dnnl_OIdhw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw2i8o16i4o,
    dnnl_IOdhw2i8o16i4o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    IOdhw2i8o16i2o,
    dnnl_IOdhw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw2o8i16o2i,
    dnnl_gOIw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOw2i8o16i2o,
    dnnl_gIOw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOhw2i8o16i2o,
    dnnl_gIOhw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gIOdhw2i8o16i2o,
    dnnl_gIOdhw2i8o16i2o,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2o8i16o2i,
    dnnl_gOIhw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIdhw2o8i16o2i,
    dnnl_gOIdhw2o8i16o2i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIw2o8i16o4i,
    dnnl_gOIw2o8i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
impl_format_tag!(
    gOIhw2o8i16o4i,
    dnnl_gOIhw2o8i16o4i,
    3,
    "3D CNN activations tensor blocked by channels with block size 8,\\n an alias to #dnnl_aBc8b"
);
