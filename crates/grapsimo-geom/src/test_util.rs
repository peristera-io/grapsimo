use crate::ApproxEq;

#[track_caller]
pub fn assert_approx_eq_impl<T: ApproxEq + std::fmt::Debug>(
    a: T,
    b: T,
    msg: Option<std::fmt::Arguments<'_>>,
) {
    match msg {
        None => assert!(
            a.approx_eq(b),
            "assert_approx_eq failed\n left: {:?}\n, right: {:?}",
            a,
            b
        ),
        Some(m) => assert!(
            a.approx_eq(b),
            "{} \n assert_approx_eq failed\n left: {:?}\n, right: {:?}",
            m,
            a,
            b
        ),
    }
}

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        $crate::test_util::assert_approx_eq_impl($a, $b, None)
    };
    ($a:expr, $b:expr, $($arg:tt)+) => {
        $crate::test_util::assert_approx_eq_impl($a, $b, Some(format_args!($($arg)+)))
    };
}
pub(crate) use assert_approx_eq;
