#include "LKRnn.h"
#include "nn_weights.h"
#include <cmath>
#include <vector>

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

LKRnn::LKRnn() : LKRv3("lkrnn") {
    x_mean = NNWeights::x_mean;
    x_std  = NNWeights::x_std;
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

    // ── 1. Скалярні змінні першої групи ───────────────────────────
    float m_lpj1 = (vecLepP + j1).M();
    float m_lmj2 = (vecLepM + j2).M();
    float ht = vecLepM.Pt() + vecLepP.Pt() + j1.Pt() + j2.Pt()
               + std::sqrt(metPx*metPx + metPy*metPy);

    // ── 2. Нові змінні: LKRv3-подібні фізичні constraints ────────────────────
    TLorentzVector llbar = vecLepM + vecLepP;
    float llbar_m   = llbar.M();
    float llbar_e   = llbar.E();
    float llbar_pz  = llbar.Pz();

    float ep = llbar_e + llbar_pz;
    float em = llbar_e - llbar_pz;
    float llbar_rap = (ep > 0.f && em > 0.f) ? 0.5f * std::log(ep / em) : 0.f;

    float met_pt  = std::sqrt(metPx*metPx + metPy*metPy);
    float mt_nunu = std::sqrt(llbar_m*llbar_m + met_pt*met_pt);
    float pz_nunu_lkr = mt_nunu * std::sinh(llbar_rap);
    float e_nunu_lkr  = mt_nunu * std::cosh(llbar_rap);

    float llnn_px = llbar.Px() + metPx;
    float llnn_py = llbar.Py() + metPy;
    float llnn_pz = llbar_pz   + pz_nunu_lkr;
    float llnn_e  = llbar_e    + e_nunu_lkr;
    float llnn_m2 = llnn_e*llnn_e - llnn_px*llnn_px - llnn_py*llnn_py - llnn_pz*llnn_pz;
    float llnn_m  = llnn_m2 > 0.f ? std::sqrt(llnn_m2) : 0.f;

    const float mw = 80.4f;
    float llnn_e_corr  = llnn_e;
    float llnn_pz_corr = llnn_pz;
    if (llnn_m < 2.0f * mw) {
        float llnn_pt2 = llnn_px*llnn_px + llnn_py*llnn_py;
        float ep2 = llnn_e + llnn_pz;
        float em2 = llnn_e - llnn_pz;
        float llnn_rap = (ep2 > 0.f && em2 > 0.f) ? 0.5f * std::log(ep2 / em2) : 0.f;
        llnn_e_corr  = std::sqrt(4.f*mw*mw + llnn_pt2) * std::cosh(llnn_rap);
        llnn_pz_corr = llnn_e_corr * std::tanh(llnn_rap);
    }

    float tt_px_lkr = llnn_px      + j1.Px() + j2.Px();
    float tt_py_lkr = llnn_py      + j1.Py() + j2.Py();
    float tt_pz_lkr = llnn_pz_corr + j1.Pz() + j2.Pz();
    float tt_e_lkr  = llnn_e_corr  + j1.E()  + j2.E();
    float tt_m2_lkr = tt_e_lkr*tt_e_lkr - tt_px_lkr*tt_px_lkr - tt_py_lkr*tt_py_lkr - tt_pz_lkr*tt_pz_lkr;

    float mtt_lkrv3_val = tt_m2_lkr > 0.f ? std::sqrt(tt_m2_lkr) : 300.f;
    float pttt_lkr      = std::sqrt(tt_px_lkr*tt_px_lkr + tt_py_lkr*tt_py_lkr);
    float phitt_lkr     = std::atan2(tt_py_lkr, tt_px_lkr);
    float ep_lkr        = tt_e_lkr + tt_pz_lkr;
    float em_lkr        = tt_e_lkr - tt_pz_lkr;
    float ytt_lkr       = (ep_lkr > 0.f && em_lkr > 0.f) ? 0.5f * std::log(ep_lkr / em_lkr) : 0.f;

    float log_mtt_lkrv3   = std::log(std::max(mtt_lkrv3_val, 300.f));
    float log_pttt_lkrv3  = std::log(pttt_lkr + 1.0f);

    // ── 3. Вхідний вектор (26 ОЗНАК) ───────────────────────────
    std::vector<float> x = {
        (float)vecLepM.E(),  (float)vecLepM.Px(), (float)vecLepM.Py(), (float)vecLepM.Pz(),
        (float)vecLepP.E(),  (float)vecLepP.Px(), (float)vecLepP.Py(), (float)vecLepP.Pz(),
        (float)j1.E(),  (float)j1.Px(), (float)j1.Py(), (float)j1.Pz(),
        (float)j2.E(),  (float)j2.Px(), (float)j2.Py(), (float)j2.Pz(),
        metPx, metPy,
        m_lpj1, m_lmj2, ht,
        llbar_m, llbar_rap, mt_nunu, pz_nunu_lkr, llnn_m
    };

    // ── 4. Нормалізація входу ─────────────────────────────────────────────────
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (x[i] - x_mean[i]) / (x_std[i] + 1e-8f);

    // ── 5. Інференс (ТІЛЬКИ 2 БЛОКИ!) ───────────────────────────
    using namespace NNWeights;
    auto h = linear_layer(x, input_proj_0_weight, input_proj_0_bias);
    h = layer_norm(h, input_proj_1_weight, input_proj_1_bias);
    silu_inplace(h);

    h = res_block(h, blocks_0_net_0_weight, blocks_0_net_0_bias, blocks_0_net_1_weight, blocks_0_net_1_bias, blocks_0_net_3_weight, blocks_0_net_3_bias, blocks_0_net_4_weight, blocks_0_net_4_bias);
    h = res_block(h, blocks_1_net_0_weight, blocks_1_net_0_bias, blocks_1_net_1_weight, blocks_1_net_1_bias, blocks_1_net_3_weight, blocks_1_net_3_bias, blocks_1_net_4_weight, blocks_1_net_4_bias);
    // ДОДАНО ТРЕТІЙ БЛОК:
    h = res_block(h, blocks_2_net_0_weight, blocks_2_net_0_bias, blocks_2_net_1_weight, blocks_2_net_1_bias, blocks_2_net_3_weight, blocks_2_net_3_bias, blocks_2_net_4_weight, blocks_2_net_4_bias);

    auto out = linear_layer(h, head_weight, head_bias);

    // ── 6. ВІДНОВЛЕННЯ ІЗ ЗАЛИШКІВ ──────────────────────────────────────────
    float d_log_mtt  = out[0];
    float d_log_pttt = out[1];
    float d_ytt      = out[2];

    const float mtt   = std::exp(log_mtt_lkrv3 + d_log_mtt);
    const float pttt  = std::exp(log_pttt_lkrv3 + d_log_pttt) - 1.0f;
    const float ytt   = ytt_lkr + d_ytt;
    const float phitt = phitt_lkr; 

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