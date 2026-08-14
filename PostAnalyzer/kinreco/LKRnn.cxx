#include "LKRnn.h"
#include "nn_weights.h"      // namespace NNWeights      (детекторна модель)
#include "nn_weights_gen.h"  // namespace NNWeights_gen  (генераторна модель)
#include <cmath>
#include <vector>

// заповнити NNParams вказівниками на ваги заданого namespace
#define FILL_NNPARAMS(P, NS) do {                                             \
    P.ip0w = &NS::input_proj_0_weight; P.ip0b = &NS::input_proj_0_bias;       \
    P.ip1w = &NS::input_proj_1_weight; P.ip1b = &NS::input_proj_1_bias;       \
    P.b0_0w = &NS::blocks_0_net_0_weight; P.b0_0b = &NS::blocks_0_net_0_bias; \
    P.b0_1w = &NS::blocks_0_net_1_weight; P.b0_1b = &NS::blocks_0_net_1_bias; \
    P.b0_3w = &NS::blocks_0_net_3_weight; P.b0_3b = &NS::blocks_0_net_3_bias; \
    P.b0_4w = &NS::blocks_0_net_4_weight; P.b0_4b = &NS::blocks_0_net_4_bias; \
    P.b1_0w = &NS::blocks_1_net_0_weight; P.b1_0b = &NS::blocks_1_net_0_bias; \
    P.b1_1w = &NS::blocks_1_net_1_weight; P.b1_1b = &NS::blocks_1_net_1_bias; \
    P.b1_3w = &NS::blocks_1_net_3_weight; P.b1_3b = &NS::blocks_1_net_3_bias; \
    P.b1_4w = &NS::blocks_1_net_4_weight; P.b1_4b = &NS::blocks_1_net_4_bias; \
    P.b2_0w = &NS::blocks_2_net_0_weight; P.b2_0b = &NS::blocks_2_net_0_bias; \
    P.b2_1w = &NS::blocks_2_net_1_weight; P.b2_1b = &NS::blocks_2_net_1_bias; \
    P.b2_3w = &NS::blocks_2_net_3_weight; P.b2_3b = &NS::blocks_2_net_3_bias; \
    P.b2_4w = &NS::blocks_2_net_4_weight; P.b2_4b = &NS::blocks_2_net_4_bias; \
    P.hw = &NS::head_weight; P.hb = &NS::head_bias;                           \
} while(0)

static std::vector<float> linear_layer(
    const std::vector<float>& in, const std::vector<float>& w, const std::vector<float>& b)
{
    int out_size = b.size(), in_size = in.size();
    std::vector<float> out(out_size);
    for (int j = 0; j < out_size; ++j) {
        float sum = b[j];
        for (int i = 0; i < in_size; ++i) sum += in[i] * w[j * in_size + i];
        out[j] = sum;
    }
    return out;
}

static std::vector<float> layer_norm(
    const std::vector<float>& x, const std::vector<float>& gamma, const std::vector<float>& beta)
{
    float mean = 0.f;
    for (float v : x) mean += v;
    mean /= x.size();
    float var = 0.f;
    for (float v : x) var += (v - mean)*(v - mean);
    var /= x.size();
    float inv_std = 1.f / std::sqrt(var + 1e-5f);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    return out;
}

static void silu_inplace(std::vector<float>& x) {
    for (float& v : x) v = v / (1.f + std::exp(-v));
}

static std::vector<float> res_block(
    const std::vector<float>& x,
    const std::vector<float>& w1, const std::vector<float>& b1,
    const std::vector<float>& ln1_w, const std::vector<float>& ln1_b,
    const std::vector<float>& w2, const std::vector<float>& b2,
    const std::vector<float>& ln2_w, const std::vector<float>& ln2_b)
{
    auto h = linear_layer(x, w1, b1);
    h = layer_norm(h, ln1_w, ln1_b);
    silu_inplace(h);
    h = linear_layer(h, w2, b2);
    h = layer_norm(h, ln2_w, ln2_b);
    for (size_t i = 0; i < h.size(); ++i) h[i] += x[i];
    silu_inplace(h);
    return h;
}

LKRnn::LKRnn(bool useGen) : LKRv3("lkrnn") {
    if (useGen) {
        x_mean = NNWeights_gen::x_mean;
        x_std  = NNWeights_gen::x_std;
        FILL_NNPARAMS(_p, NNWeights_gen);
    } else {
        x_mean = NNWeights::x_mean;
        x_std  = NNWeights::x_std;
        FILL_NNPARAMS(_p, NNWeights);
    }
}

std::vector<TLorentzVector> LKRnn::reconstruct(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const std::vector<TLorentzVector>& vecJets,
    Float_t* jetBTagDiscr, const double bTagDiscrL,
    const Float_t metPx, const Float_t metPy)
{
    std::vector<TLorentzVector> solution;

    TLorentzVector j1, j2;
    if (!selectBestJets(vecLepM, vecLepP, vecJets, jetBTagDiscr, bTagDiscrL, j1, j2))
        return solution;

    // ── Вхід мережі: 26 ознак (ЄДИНА реалізація — LKRv3::computeFeatures) ──────
    std::vector<float> x = computeFeatures(vecLepM, vecLepP, j1, j2, metPx, metPy);

    // ── База LKRv3, яку мережа коригує (той самий solve, що й у LKRv3) ─────────
    TLorentzVector ttbar_lkrv3 = solve(vecLepM, vecLepP, j1, j2, metPx, metPy);
    float mtt_lkrv3_val = ttbar_lkrv3.M();
    float pttt_lkr      = ttbar_lkrv3.Pt();
    float phitt_lkr     = ttbar_lkrv3.Phi();
    // рапідність із захистом (TLorentzVector::Rapidity() дає inf/nan при E-Pz<=0)
    float ep_lkr = ttbar_lkrv3.E() + ttbar_lkrv3.Pz();
    float em_lkr = ttbar_lkrv3.E() - ttbar_lkrv3.Pz();
    float ytt_lkr = (ep_lkr > 0.f && em_lkr > 0.f) ? 0.5f * std::log(ep_lkr / em_lkr) : 0.f;
    float log_mtt_lkrv3  = std::log(std::max(mtt_lkrv3_val, 300.f));
    float log_pttt_lkrv3 = std::log(pttt_lkr + 1.0f);

    // ── 4. Нормалізація входу ─────────────────────────────────────────────────
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (x[i] - x_mean[i]) / (x_std[i] + 1e-8f);

    // ── 5. Інференс (3 БЛОКИ) з ваг обраної моделі (_p) ───────────────────────
    auto h = linear_layer(x, *_p.ip0w, *_p.ip0b);
    h = layer_norm(h, *_p.ip1w, *_p.ip1b);
    silu_inplace(h);

    h = res_block(h, *_p.b0_0w, *_p.b0_0b, *_p.b0_1w, *_p.b0_1b, *_p.b0_3w, *_p.b0_3b, *_p.b0_4w, *_p.b0_4b);
    h = res_block(h, *_p.b1_0w, *_p.b1_0b, *_p.b1_1w, *_p.b1_1b, *_p.b1_3w, *_p.b1_3b, *_p.b1_4w, *_p.b1_4b);
    h = res_block(h, *_p.b2_0w, *_p.b2_0b, *_p.b2_1w, *_p.b2_1b, *_p.b2_3w, *_p.b2_3b, *_p.b2_4w, *_p.b2_4b);

    auto out = linear_layer(h, *_p.hw, *_p.hb);

    // ── 6. ВІДНОВЛЕННЯ З ФІЗИЧНИМ ОБМЕЖЕННЯМ (TANH) ────────────────────
    float d_log_mtt  = std::tanh(out[0]) * 0.05f; // <--- ТУТ ТІЛЬКИ 0.05f
    float d_log_pttt = std::tanh(out[1]) * 0.20f;
    float d_ytt      = std::tanh(out[2]) * 0.05f;

    const float mtt   = std::exp(log_mtt_lkrv3 + d_log_mtt);
    const float pttt  = std::exp(log_pttt_lkrv3 + d_log_pttt) - 1.0f;
    const float ytt   = ytt_lkr + d_ytt;
    const float phitt = phitt_lkr;

    if (std::isnan(mtt) || mtt <= 0.f || std::isnan(pttt) || pttt < 0.f) return solution;
    if (mtt <= 0.f || pttt < 0.f) return solution;

    // ── 7. Побудова ttbar TLorentzVector ─────────────────────────────────────
    const float px = pttt * std::cos(phitt);
    const float py = pttt * std::sin(phitt);
    const float mt = std::sqrt(mtt*mtt + pttt*pttt);
    const float pz = mt * std::sinh(ytt);
    const float E  = mt * std::cosh(ytt);

    solution.resize(3);
    solution[2].SetPxPyPzE(px, py, pz, E);
    solution[0].SetPxPyPzE(px/2.f, py/2.f, pz/2.f, E/2.f);
    solution[1].SetPxPyPzE(px/2.f, py/2.f, pz/2.f, E/2.f);
    return solution;
}