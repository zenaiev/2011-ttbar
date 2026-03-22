import ROOT
from array import array

# Вхід
f_main = ROOT.TFile("ttbar_output_3_main.root", "READ")
t_main = f_main.Get("ttbarTree")

f_rej  = ROOT.TFile("ttbar_output_3_rejected.root", "READ")
t_rej  = f_rej.Get("ttbarTree")

if t_main.GetEntries() != t_rej.GetEntries():
    raise RuntimeError("ERROR: main та rejected мають різну кількість подій!")

# Вихідний файл (новий, правильний)
f_out = ROOT.TFile("ttbar_output_3_main_fixed.root", "RECREATE")

# Клонуємо дерево без подій
t_out = t_main.CloneTree(0)

# Знаходимо lkrv2-гілки
branches_to_copy = [
    br.GetName() for br in t_rej.GetListOfBranches()
    if "lkrv2" in br.GetName()
]

# Підготовка буферів
rej_buf = {}
new_buf = {}

for br in branches_to_copy:
    rej_buf[br] = array('f', [0.0])
    t_rej.SetBranchAddress(br, rej_buf[br])

    new_name = br + "_rejected"
    new_buf[new_name] = array('f', [0.0])
    t_out.Branch(new_name, new_buf[new_name], new_name + "/F")

# Копіюємо
N = t_main.GetEntries()
for i in range(N):
    t_main.GetEntry(i)
    t_rej.GetEntry(i)

    # Записуємо нові значення
    for br in branches_to_copy:
        new_name = br + "_rejected"
        new_buf[new_name][0] = rej_buf[br][0]

    t_out.Fill()

# Запис
t_out.Write()
f_out.Close()
f_rej.Close()
f_main.Close()
