#include "LKRnn.h"
#include "TMVA/RSofieReader.hxx"
#include <TInterpreter.h>
#include <TSystem.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>

// Мережа (ONNX: нормування входу, шари, tanh·масштаб) читається під час запуску через ROOT TMVA SOFIE:
// RSofieReader генерує з ONNX C++-код і компілює його інтерпретатором.
//  * Згенерований код має простір імен за іменем файлу й пишеться в поточну папку, тож кожна модель
//    завантажується в окремій тимчасовій папці з унікальним іменем (det- і gen-модель, паралельні
//    процеси run_full_parallel.sh не конфліктують).
//  * Кожна модель компілюється один раз на процес і перевикористовується (два проходи eventreco()).
//  * Тимчасові файли видаляються лише при завершенні процесу: якщо видалити їх одразу, наступний
//    згенерований header може отримати той самий inode, і інтерпретатор вважатиме його вже підключеним.
namespace {

const char* kModelFile = "solve_nn.onnx";   // файл моделі в її папці (пишуть train_solve_nn.py / export_onnx.py)

struct SofieCache {
    std::map<std::string, std::shared_ptr<TMVA::Experimental::RSofieReader>> readers;
    std::vector<std::string> tmpDirs;
    int loads = 0;
};
// навмисно не звільняється: скомпільований код моделей живе в інтерпретаторі до кінця процесу
SofieCache* gSofie = new SofieCache;

void RemoveSofieTmpDirs() {
    for (const auto& d : gSofie->tmpDirs) {
        std::vector<std::string> files;
        if (DIR* dir = opendir(d.c_str())) {
            while (dirent* e = readdir(dir)) {
                const std::string n(e->d_name);
                if (n != "." && n != "..") files.push_back(d + "/" + n);
            }
            closedir(dir);
        }
        for (const auto& f : files) std::remove(f.c_str());
        rmdir(d.c_str());
    }
}

std::shared_ptr<TMVA::Experimental::RSofieReader> LoadSofieModel(const std::string& absPath) {
    auto cached = gSofie->readers.find(absPath);
    if (cached != gSofie->readers.end()) return cached->second;
    if (gSofie->loads == 0) std::atexit(RemoveSofieTmpDirs);

    std::string tmpl = std::string(gSystem->TempDirectory()) + "/lkrnn_sofie_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) {
        perror("[E] LKRnn: mkdtemp");
        exit(1);
    }
    const std::string tmpDir(buf.data());
    const std::string name = "lkrnn_" + std::to_string(getpid()) + "_" + std::to_string(gSofie->loads++);
    if (gSystem->CopyFile(absPath.c_str(), (tmpDir + "/" + name + ".onnx").c_str()) != 0) {
        fprintf(stderr, "[E] LKRnn: не вдалося скопіювати %s у %s\n", absPath.c_str(), tmpDir.c_str());
        exit(1);
    }

    const std::string cwd = gSystem->WorkingDirectory();
    const std::string log = tmpDir + "/load.log";
    fflush(stdout);
    fflush(stderr);
    // інтерпретатор шукає #include відносно папки, де він ініціалізувався, — додаємо тимчасову папку
    gInterpreter->AddIncludePath(tmpDir.c_str());
    gSystem->ChangeDirectory(tmpDir.c_str());
    gSystem->RedirectOutput(log.c_str(), "w");      // SOFIE друкує службові рядки під час завантаження
    std::shared_ptr<TMVA::Experimental::RSofieReader> reader;
    std::string loadError;
    try {
        reader = std::make_shared<TMVA::Experimental::RSofieReader>(name + ".onnx");
    } catch (const std::exception& e) {
        loadError = e.what();
    }
    gSystem->RedirectOutput(nullptr);
    gSystem->ChangeDirectory(cwd.c_str());

    std::vector<float> probe;
    if (loadError.empty() && reader) {
        try {
            probe = reader->Compute(std::vector<float>(26, 0.f));
        } catch (const std::exception& e) {
            loadError = e.what();
        }
    }
    if (probe.size() != 3) {
        fprintf(stderr, "[E] LKRnn: не вдалося завантажити модель %s через SOFIE (тимчасові файли: %s)\n%s\nлог завантаження:\n",
                absPath.c_str(), tmpDir.c_str(), loadError.c_str());
        std::ifstream f(log);
        std::cerr << f.rdbuf() << std::endl;
        exit(1);
    }
    gSofie->tmpDirs.push_back(tmpDir);
    gSofie->readers[absPath] = reader;
    printf("[I] LKRnn: модель %s завантажено (SOFIE)\n", absPath.c_str());
    return reader;
}

}  // namespace

LKRnn::LKRnn(const std::string& modelDir) : LKRv3("lkrnn") {
    const std::string src = modelDir + "/" + kModelFile;
    char* absBuf = realpath(src.c_str(), nullptr);
    if (!absBuf) {
        fprintf(stderr, "[E] LKRnn: немає моделі %s (експорт: python export_onnx.py --model-dir %s)\n",
                src.c_str(), modelDir.c_str());
        exit(1);
    }
    const std::string absPath(absBuf);
    free(absBuf);
    _reader = LoadSofieModel(absPath);
}

LKRnn::~LKRnn() = default;

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
    // база LKRv3 без обрізання: раніше max(m, 300) зводив усі події з m < 300 ГеВ до однієї бази,
    // і з граничною поправкою +5% вони давали пік рівно при 300·e^0.05 = 315.38 ГеВ
    if (!(mtt_lkrv3_val > 0.f)) return solution;
    float log_mtt_lkrv3  = std::log(mtt_lkrv3_val);
    float log_pttt_lkrv3 = std::log(pttt_lkr + 1.0f);

    // ── 4–6. Мережа: нормування входу, шари й tanh·масштаб (0.05 / 0.20 / 0.05) — усе всередині ONNX ──
    const std::vector<float> d = _reader->Compute(x);
    const float d_log_mtt  = d[0];
    const float d_log_pttt = d[1];
    const float d_ytt      = d[2];

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