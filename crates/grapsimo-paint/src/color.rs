/// RGBA Color
/// represented internally by [u8;4]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Color([u8; 4]);

impl Color {
    pub const WHITE: Color = Self([255, 255, 255, 255]);
    pub const BLACK: Color = Self([0, 0, 0, 255]);
    pub const TRANSPARENT: Color = Self([0, 0, 0, 0]);

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    pub fn r(self) -> u8 {
        self.0[0]
    }
    pub fn g(self) -> u8 {
        self.0[1]
    }
    pub fn b(self) -> u8 {
        self.0[2]
    }
    pub fn a(self) -> u8 {
        self.0[3]
    }

    pub fn to_premul_color(self) -> PremulColor {
        PremulColor::from_color(self)
    }
}

///
/// Premultiplied color (by alpha channel)
/// represented internally by [u8;4]

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PremulColor([u8; 4]);

impl PremulColor {
    pub const fn from_array(a: [u8; 4]) -> Self {
        debug_assert!(a[0] <= a[3] && a[1] <= a[3] && a[2] <= a[3]);
        Self(a)
    }

    ///returns as u32 AABBGGRR
    pub const fn to_u32(self) -> u32 {
        u32::from_le_bytes(self.0)
    }

    ///expects a u32 as AABBGGRR
    pub const fn from_u32(value: u32) -> Self {
        Self(value.to_le_bytes())
    }

    pub fn to_color(self) -> Color {
        Color::rgba(
            Self::color_from_premul_color(self.r(), self.a()),
            Self::color_from_premul_color(self.g(), self.a()),
            Self::color_from_premul_color(self.b(), self.a()),
            self.a(),
        )
    }

    pub fn from_color(c: Color) -> Self {
        Self([
            Self::premul_color_from_color(c.r(), c.a()),
            Self::premul_color_from_color(c.g(), c.a()),
            Self::premul_color_from_color(c.b(), c.a()),
            c.a(),
        ])
    }

    pub fn r(self) -> u8 {
        self.0[0]
    }
    pub fn g(self) -> u8 {
        self.0[1]
    }
    pub fn b(self) -> u8 {
        self.0[2]
    }
    pub fn a(self) -> u8 {
        self.0[3]
    }

    fn color_from_premul_color(premul_color: u8, alpha: u8) -> u8 {
        // if this ever gets on the hotpath - consider using a lookup table
        // for the division
        if alpha == 0 {
            0
        } else {
            (premul_color as u32 * 255 / alpha as u32) as u8
        }
    }

    fn premul_color_from_color(color: u8, alpha: u8) -> u8 {
        ((color as u32 * alpha as u32 + 127) / 255) as u8
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn premul_fast() {
        for i in 0..u8::MAX {
            for j in 0..u8::MAX {
                assert_eq!(
                    ((i as f64 * j as f64) / 255.0).round() as u8,
                    PremulColor::premul_color_from_color(i, j),
                    "color={i} alpha={j}"
                )
            }
        }
    }
}
