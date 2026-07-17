import math, random  # run with: pip install fastapi httpx --break-system-packages
from wind_backend import weibull_mle, build_rose, icing_criterion, hysteresis_loss, sector_index

random.seed(42)

def weibull_sample(A, k, n):
    # inverse-CDF sampling: v = A * (-ln(1-u))^(1/k)
    return [A * (-math.log(1.0 - random.random())) ** (1.0 / k) for _ in range(n)]

print("=== 1. Weibull MLE recovery (does the fit find known A,k?) ===")
for (A_true, k_true) in [(8.0, 2.0), (10.5, 2.4), (6.2, 1.7), (11.0, 2.9)]:
    s = weibull_sample(A_true, k_true, 175000)   # ~20 yrs hourly
    A, k = weibull_mle(s)
    print(f"  true A={A_true:5.2f} k={k_true:4.2f}  ->  fit A={A:5.2f} k={k:4.2f}"
          f"   err {100*(A-A_true)/A_true:+5.2f}% / {100*(k-k_true)/k_true:+5.2f}%")

print("\n=== 2. Calm-threshold bias (the documented caveat, quantified) ===")
s = weibull_sample(8.0, 2.0, 200000)
for thr in [0.0001, 0.5, 1.0, 2.0]:
    A, k = weibull_mle(s, calm_threshold=thr)
    print(f"  threshold {thr:5.3f} m/s -> A={A:5.3f} k={k:5.3f}")

print("\n=== 3. Sector binning edges ===")
for d in [0, 14.9, 15.1, 29, 345, 359.9, 360, 375]:
    print(f"  {d:6.1f} deg -> sector {sector_index(d)}")

print("\n=== 4. Rose on synthetic directional data ===")
sp, di = [], []
for _ in range(120000):
    d = random.gauss(240, 45) % 360          # SW-prevailing
    sp.append(weibull_sample(9.0 if 200 < d < 280 else 6.5, 2.1, 1)[0])
    di.append(d)
r = build_rose(sp, di)
tot = sum(r["frequency_pct"])
print(f"  freq sums to {tot:.3f}%  (must be 100)")
print(f"  all sectors fitted: {all(r['sectors_fitted'])}")
peak = max(range(12), key=lambda i: r["frequency_pct"][i])
print(f"  peak sector = {peak} ({peak*30} deg), A={r['weibull_A'][peak]:.2f}  <- expect ~240 deg, high A")

print("\n=== 5. Hysteresis (chronological, cannot come from Weibull) ===")
# construct: rises through cut-out, then decays slowly back down
series = [20,24,26,27,24,23.5,23,22.5,22,21.5,21,20.5,19,18,10]
h = hysteresis_loss(series, cut_out=25, restart=22)
print(f"  {h}")
print("  expect: hours at 24,23.5,23,22.5 lost (in-limits but post-trip, above restart) = 4")

print("\n=== 6. Hysteresis sanity: never-trips series must give zero ===")
print(f"  {hysteresis_loss([5,8,12,15,20,24,10], cut_out=25, restart=22)['lost_hours']} (expect 0)")

print("\n=== 7. Icing criterion ===")
t = [-5]*100 + [3]*900     # 10% of hours below zero
rh = [98]*50 + [80]*50 + [98]*900   # only 50 of the cold hours are also humid
print(f"  {icing_criterion(t, rh, hub_height=100)}")
print("  expect ~5% (50/1000), tripped=True")
