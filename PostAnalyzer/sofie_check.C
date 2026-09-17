// Перевірка, що ROOT TMVA SOFIE і BLAS працюють: завантажує ONNX-модель і рахує один вихід.
#include "TMVA/RSofieReader.hxx"
void sofie_check(const char* onnx) {
   TMVA::Experimental::RSofieReader r(onnx);
   auto y = r.Compute(std::vector<float>(26, 0.f));
   printf("\nSOFIE і BLAS працюють: модель повернула %zu числа (%g, %g, %g)\n", y.size(), y[0], y[1], y[2]);
}
