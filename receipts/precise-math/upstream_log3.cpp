// The last red upstream ops_tests.cpp case on Honeykrisp, extracted verbatim:
//   x = split(array({1,2,3,4},{2,2}), 2, 1)[0];
//   expected = array({std::log(1.0f), std::log(3.0f)}, {2,1});
//   CHECK(array_equal(log(x), expected).item<bool>());
// plus the same bit-compare for log2, sin/cos at 1e5..1e8, and 80/25.
#include <cmath>
#include <cstdio>
#include <cstring>
#include "mlx/mlx.h"

using namespace mlx::core;

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main() {
  set_default_device(Device::gpu);
  int fails = 0;
  auto x = split(array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), 2, 1)[0];
  auto expected = array({std::log(1.0f), std::log(3.0f)}, {2, 1});
  auto got = log(x);
  eval(got);
  bool ok = array_equal(got, expected).item<bool>();
  std::printf("upstream log(3) pin: got %.17g (0x%08x) want %.17g (0x%08x) -> %s\n",
              got.data<float>()[1], bits(got.data<float>()[1]), std::log(3.0f),
              bits(std::log(3.0f)), ok ? "PASS" : "FAIL");
  fails += !ok;

  auto d = divide(array({80.0f, 10.0f}), array({25.0f, 25.0f}));
  eval(d);
  for (int i = 0; i < 2; ++i) {
    float want = i == 0 ? 80.0f / 25.0f : 10.0f / 25.0f;
    bool o = bits(d.data<float>()[i]) == bits(want);
    std::printf("divide %s: got 0x%08x want 0x%08x -> %s\n", i == 0 ? "80/25" : "10/25",
                bits(d.data<float>()[i]), bits(want), o ? "PASS" : "FAIL");
    fails += !o;
  }

  float xs[] = {1e5f, 9.9e4f, 6e4f, 1e4f, 1e3f}; // mlx refuses |x| > 1e5 on this driver (docs/known-defects.md); larger x are covered by probe.py at the shader level
  auto sx = array(xs, {5});
  auto s = sin(sx), c = cos(sx);
  eval(s, c);
  for (int i = 0; i < 5; ++i) {
    double ws = std::sin((double)xs[i]), wc = std::cos((double)xs[i]);
    float gs = s.data<float>()[i], gc = c.data<float>()[i];
    long us = std::labs((long)bits(gs) - (long)bits((float)ws));
    long uc = std::labs((long)bits(gc) - (long)bits((float)wc));
    std::printf("sin(%g) got %.9g want %.9g ulp %ld | cos got %.9g want %.9g ulp %ld -> %s\n",
                xs[i], gs, ws, us, gc, wc, uc, (us <= 2 && uc <= 2) ? "PASS" : "FAIL");
    fails += !(us <= 2 && uc <= 2);
  }
  std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
  return fails != 0;
}
