//! Bessel functions J0, J1, and the positive zeros of J0.
//!
//! j0()/j1() port the Cephes Math Library's rational-Chebyshev
//! approximations (Moshier, 2000; the same algorithm underlying many libm
//! implementations) — the standard library has none, and this keeps the
//! model dependency-free. j0_zero() seeds the n-th root with McMahon's
//! asymptotic expansion, then polishes it with Newton's method (J0' = -J1).

const PIO4: f64 = 0.78539816339744830962;
const SQ2OPI: f64 = 0.79788456080286535588;

fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// Same as polevl, but the leading (implicit) coefficient of the highest
/// power is 1.0 and omitted from `coef`.
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(1.0, |acc, &c| acc * x + c)
}

const J0_RP: [f64; 4] = [
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
];
const J0_RQ: [f64; 8] = [
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18,
];
const J0_PP: [f64; 7] = [
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1,
];
const J0_PQ: [f64; 7] = [
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0,
];
const J0_QP: [f64; 8] = [
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
];
const J0_QQ: [f64; 7] = [
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2,
];
const J0_DR1: f64 = 5.78318596294678452118E0;
const J0_DR2: f64 = 3.04712623436620863991E1;

/// Bessel function of the first kind, order 0.
pub fn j0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 5.0 {
        let z = x * x;
        if x < 1.0e-5 {
            return 1.0 - z / 4.0;
        }
        (z - J0_DR1) * (z - J0_DR2) * polevl(z, &J0_RP) / p1evl(z, &J0_RQ)
    } else {
        let w = 5.0 / x;
        let q = 25.0 / (x * x);
        let p = polevl(q, &J0_PP) / polevl(q, &J0_PQ);
        let qf = polevl(q, &J0_QP) / p1evl(q, &J0_QQ);
        let xn = x - PIO4;
        (p * xn.cos() - w * qf * xn.sin()) * SQ2OPI / x.sqrt()
    }
}

const J1_RP: [f64; 4] = [
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
];
const J1_RQ: [f64; 8] = [
    6.20836478118054335476E2,
    2.56987256757748830383E5,
    8.35146791431949253037E7,
    2.21511595479792499675E10,
    4.74914122079991414898E12,
    7.84369607876235854894E14,
    8.95222336184627338078E16,
    5.32278620332680085395E18,
];
const J1_PP: [f64; 7] = [
    7.62125616208173112003E-4,
    7.31397056940917570436E-2,
    1.12719608129684925192E0,
    5.11207951146807644818E0,
    8.42404590141772420927E0,
    5.21451598682361504063E0,
    1.00000000000000000254E0,
];
const J1_PQ: [f64; 7] = [
    5.71323128072548699714E-4,
    6.88455908754495404082E-2,
    1.10514232634061696926E0,
    5.07386386128601488557E0,
    8.39985554327604159757E0,
    5.20982848682361821619E0,
    9.99999999999999997461E-1,
];
const J1_QP: [f64; 8] = [
    5.10862594750176621635E-2,
    4.98213872951233449420E0,
    7.58238284132545283818E1,
    3.66779609360150777800E2,
    7.10856304998926107277E2,
    5.97489612400613639965E2,
    2.11688757100572135698E2,
    2.52070205858023719784E1,
];
const J1_QQ: [f64; 7] = [
    7.42373277035675149943E1,
    1.05644886038262816351E3,
    4.98641058337653607651E3,
    9.56231892404756170795E3,
    7.99704160447350683650E3,
    2.82619278517639096600E3,
    3.36093607810698293419E2,
];
const J1_Z1: f64 = 1.46819706421238932572E1;
const J1_Z2: f64 = 4.92184563216946036703E1;
const THPIO4: f64 = 2.35619449019234492885;

/// Bessel function of the first kind, order 1.
pub fn j1(x: f64) -> f64 {
    let w = x.abs();
    if w <= 5.0 {
        let z = x * x;
        let mut w = polevl(z, &J1_RP) / p1evl(z, &J1_RQ);
        w = w * x * (z - J1_Z1) * (z - J1_Z2);
        w
    } else {
        let w = 5.0 / x;
        let z = w * w;
        let p = polevl(z, &J1_PP) / polevl(z, &J1_PQ);
        let q = polevl(z, &J1_QP) / p1evl(z, &J1_QQ);
        let xn = x - THPIO4;
        (p * xn.cos() - w * q * xn.sin()) * SQ2OPI / x.sqrt()
    }
}

/// The n-th positive zero of J0 (j_{0,n} such that J0(j_{0,n}) = 0), n = 1,
/// 2, 3... Seeded via McMahon's asymptotic expansion (Abramowitz & Stegun
/// 9.5.12), polished with Newton's method (J0' = -J1) — the seed is within
/// ~1e-3 of the true root even at n=1, safely inside Newton's basin given
/// roots are spaced ~π apart.
pub fn j0_zero(n: u32) -> f64 {
    let beta = (n as f64 - 0.25) * std::f64::consts::PI;
    let eight_beta = 8.0 * beta;
    let mut x = beta + 1.0 / eight_beta - 124.0 / (3.0 * eight_beta.powi(3));

    for _ in 0..6 {
        x += j0(x) / j1(x);
    }
    x
}
