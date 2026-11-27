import math
import numpy as np
import matplotlib.pyplot as plt

'''
Implémentation de capnosyn —— capnogramme synthétique (courbe de CO2 expiré)
Écrit sur le modèle de ecgsyn :

function [co2, iphases] = capnosyn(sfco2, N, anoise, rrmean, rrstd, ieratio,
                                   etco2, etco2std, fico2, risetime, slope,
                                   falltime, deadspace, seed)

Sorties :
co2:      signal de pression partielle de CO2 (mmHg), un élément par échantillon
iphases:  tableau de même longueur que co2, qui marque les points clés :
          0 = début de la phase II (début de la montée expiratoire)
          1 = début de la phase III (début du plateau alvéolaire)
          2 = point EtCO2 (fin d'expiration, point le plus haut du plateau)
          3 = retour à la ligne de base (fin de la descente inspiratoire)
          -1 = point ordinaire

Entrées :
sfco2:    fréquence d'échantillonnage (Hz)
N:        nombre de cycles respiratoires à générer
anoise:   amplitude du bruit additif uniforme (mmHg)
rrmean:   fréquence respiratoire moyenne (cycles/minute)
rrstd:    écart-type de la fréquence respiratoire (cycles/minute)
ieratio:  durée d'expiration / durée d'inspiration, environ 2 en temps normal (I:E = 1:2)
etco2:    pression partielle de CO2 en fin d'expiration (mmHg), normale 35~45
etco2std: variation de l'EtCO2 d'un cycle à l'autre (mmHg)
fico2:    CO2 inspiré / ligne de base (mmHg), 0 en temps normal, > 0 en cas de réinhalation
risetime: durée de la montée de la phase II (s), environ 0,1~0,2 en temps normal,
          plus longue en cas de bronchospasme (forme en « aileron de requin »)
slope:    pente du plateau de la phase III (mmHg/s), faible en temps normal,
          plus forte en cas de pathologie obstructive
falltime: constante de temps de la descente inspiratoire de la phase 0 (s)
deadspace:fraction de la durée d'expiration pendant laquelle il n'y a pas encore de CO2
          (le gaz de l'espace mort anatomique est expiré en premier)
seed:     graine aléatoire ; une fois fixée, la courbe ne saute pas quand on bouge les curseurs

Forme d'un cycle respiratoire (d'abord l'inspiration, puis l'expiration) :

  CO2 |              Phase III      EtCO2
      |            ____________-----*
      |           /                  |
      |  Phase II/                   | Phase 0 (début de l'inspiration suivante, chute rapide)
      | ________/                    |_________
      |  Phase I (inspiration + gaz de l'espace mort)
      +----------------------------------------> t
         |<- inspiration Ti ->|<---- expiration Te ---->|
'''


def breathprocess(rrmean, rrstd, N, rng, phi=0.7):
    '''
    Génère la fréquence respiratoire (cycles/minute) de chaque cycle.
    Équivalent de rrprocess dans ecg, mais avec un processus AR(1) plus simple :
    rr[k] = rrmean + phi*(rr[k-1]-rrmean) + bruit
    Plus phi est proche de 1, plus les cycles successifs se ressemblent
    (variation plus lente).
    Le coefficient de bruit sqrt(1-phi^2) garantit que l'écart-type global vaut rrstd.
    '''
    rr = np.empty(N)
    rr[0] = rrmean + rrstd * rng.standard_normal()
    for k in range(1, N):
        rr[k] = (rrmean + phi * (rr[k - 1] - rrmean)
                 + math.sqrt(1 - phi ** 2) * rrstd * rng.standard_normal())
    return np.clip(rr, 2, 60)


def capnosyn(sfco2=100,
             N=10,
             anoise=0,
             rrmean=15,
             rrstd=1,
             ieratio=2.0,
             etco2=38,
             etco2std=0.5,
             fico2=0,
             risetime=0.15,
             slope=1.0,
             falltime=0.1,
             deadspace=0.1,
             seed=None):

    rng = np.random.default_rng(seed)
    N = int(N)

    # Durée de chaque cycle, découpée en inspiration et expiration selon le rapport I:E
    rr = breathprocess(rrmean, rrstd, N, rng)
    T = 60.0 / rr
    Ti = T / (1 + ieratio)
    Te = T - Ti

    # L'EtCO2 varie légèrement d'un cycle à l'autre
    etk = etco2 + etco2std * rng.standard_normal(N)
    etk = np.maximum(etk, fico2 + 1)

    n = int(np.floor(np.sum(T) * sfco2))
    t = np.arange(n) / sfco2
    co2 = np.full(n, float(fico2))
    iphases = -np.ones(n, dtype=int)

    def mark(time, label):
        idx = int(round(time * sfco2))
        if 0 <= idx < n:
            iphases[idx] = label

    t0 = 0.0
    prev_end = float(fico2)   # CO2 à la fin de l'expiration précédente, utilisé pour la descente
    for k in range(N):
        # ---------- Inspiration : descente phase 0 + ligne de base phase I ----------
        ins = (t >= t0) & (t < t0 + Ti[k])
        tau = t[ins] - t0
        co2[ins] = fico2 + (prev_end - fico2) * np.exp(-tau / falltime)
        if k > 0:
            mark(t0 + 3 * falltime, 3)

        # ---------- Expiration : espace mort → montée phase II → plateau phase III ----------
        te0 = t0 + Ti[k]
        ex = (t >= te0) & (t < te0 + Te[k])
        u = t[ex] - te0

        rise = min(risetime, 0.7 * Te[k])          # la montée ne peut pas durer plus que l'expiration
        mid = deadspace * Te[k] + rise / 2         # milieu de la phase II
        # Montée douce de 0 à 1, accomplie à peu près dans [mid - rise/2, mid + rise/2]
        S = 0.5 * (1 + np.tanh((u - mid) / (rise / 4)))
        # Plateau alvéolaire : montée linéaire qui atteint exactement l'EtCO2 en fin d'expiration
        A = np.maximum(etk[k] - slope * (Te[k] - u), fico2)
        co2[ex] = fico2 + (A - fico2) * S

        mark(te0 + mid - rise / 2, 0)
        mark(te0 + mid + rise / 2, 1)
        mark(te0 + Te[k] - 1 / sfco2, 2)

        S_end = 0.5 * (1 + np.tanh((Te[k] - mid) / (rise / 4)))
        prev_end = fico2 + (etk[k] - fico2) * S_end
        t0 += T[k]

    # Ajout d'un bruit de mesure uniformément distribué
    eta = 2 * rng.random(n) - 1
    co2 = np.maximum(co2 + anoise * eta, 0)

    return (co2, iphases)


PHASE_COLORS = ["#2878B5", "#2CA02C", "#D62728", "#9467BD"]
PHASE_LABELS = ["Phase II start", "Phase III start", "EtCO2", "Baseline"]


def monitor_readings(t, co2, iphases, window):
    '''Comme un moniteur, lit l'EtCO2 et la fréquence respiratoire à partir de la courbe'''
    idx = np.where((iphases == 2) & (t <= window))[0]
    if len(idx) == 0:
        return "EtCO2   -- mmHg\nRR      -- /min"
    et = np.mean(co2[idx])
    if len(idx) >= 2:
        rr = 60 / np.mean(np.diff(t[idx]))
        rr_txt = f"{rr:5.0f}"
    else:
        rr_txt = "   --"
    return f"EtCO2 {et:5.0f} mmHg\nRR    {rr_txt} /min"


def main():
    sfco2 = 100
    co2, phases = capnosyn(sfco2=sfco2, N=5, seed=0)
    t = np.arange(len(co2)) / sfco2

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, co2, color="#E0A800", linewidth=1.6, label="CO2")
    for i in range(4):
        idx = np.where(phases == i)[0]
        ax.scatter(t[idx], co2[idx], color=PHASE_COLORS[i], s=25,
                   zorder=5, label=PHASE_LABELS[i])

    ax.set_title("Synthetic Capnogram", fontsize=17, fontweight="bold", pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("CO2 (mmHg)", fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 60)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.legend(loc="upper right", ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("synthetic_capnogram.png", dpi=300, bbox_inches="tight")


def calc_N(rrmean, time_to_draw):
    # Générer quelques cycles de plus, pour que la courbe remplisse la fenêtre
    # même si la respiration ralentit
    return max(int(np.ceil(rrmean / 60 * time_to_draw * 1.3)) + 2, 3)


# Paramètres prédéfinis pour des situations cliniques courantes
PRESETS = {
    "Normal": dict(rrmean=15, etco2=38, risetime=0.15, slope=1.0,
                   fico2=0, ieratio=2.0),
    "Hyperventilation": dict(rrmean=28, etco2=25, risetime=0.15, slope=1.0,
                             fico2=0, ieratio=1.5),
    "Hypoventilation": dict(rrmean=7, etco2=60, risetime=0.15, slope=0.5,
                            fico2=0, ieratio=2.0),
    "Bronchospasm": dict(rrmean=18, etco2=45, risetime=1.0, slope=6.0,
                         fico2=0, ieratio=3.0),
    "Rebreathing": dict(rrmean=15, etco2=45, risetime=0.15, slope=1.0,
                        fico2=8, ieratio=2.0),
}


def interactive_capno():

    from matplotlib.widgets import Slider, Button, RadioButtons

    # ----------------------------------
    # Initial parameters
    # ----------------------------------

    params = {
        "time_to_draw": 20,
        "sfco2": 100,
        "rrmean": 15,
        "rrstd": 1.0,
        "ieratio": 2.0,
        "etco2": 38,
        "etco2std": 0.5,
        "fico2": 0,
        "risetime": 0.15,
        "slope": 1.0,
        "anoise": 0.0,
        "seed": 0,
    }

    def generate():
        co2, phases = capnosyn(
            sfco2=params["sfco2"],
            N=calc_N(params["rrmean"], params["time_to_draw"]),
            anoise=params["anoise"],
            rrmean=params["rrmean"],
            rrstd=params["rrstd"],
            ieratio=params["ieratio"],
            etco2=params["etco2"],
            etco2std=params["etco2std"],
            fico2=params["fico2"],
            risetime=params["risetime"],
            slope=params["slope"],
            seed=params["seed"],
        )
        t = np.arange(len(co2)) / params["sfco2"]
        return t, co2, phases

    t, co2, phases = generate()

    # ----------------------------------
    # Figure
    # ----------------------------------

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.08, bottom=0.50, right=0.78, top=0.90)
    fig.suptitle("Interactive Capnography Simulator",
                 fontsize=18, fontweight="bold")

    line, = ax.plot(t, co2, color="#E0A800", linewidth=1.6)
    ax.set_title("Synthetic Capnogram (exhaled CO2)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CO2 (mmHg)")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    markers = []
    for i in range(4):
        idx = np.where(phases == i)[0]
        markers.append(ax.scatter(t[idx], co2[idx], color=PHASE_COLORS[i],
                                  s=25, zorder=5, label=PHASE_LABELS[i]))
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    # Affichage façon moniteur
    monitor = fig.text(
        0.81, 0.78, "", fontsize=15, family="monospace",
        color="#E0A800", fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="#555555",
                  boxstyle="round,pad=0.6"),
    )

    # ----------------------------------
    # Sliders
    # ----------------------------------

    sliders = {}

    def create_slider(name, label, position, minimum, maximum):
        slider_ax = fig.add_axes(position)
        sliders[name] = Slider(slider_ax, label, minimum, maximum,
                               valinit=params[name])

    left_col = [
        ("time_to_draw", "Window (s)", 5, 60),
        ("rrmean", "Resp. rate (/min)", 4, 40),
        ("rrstd", "RR std (/min)", 0, 5),
        ("ieratio", "I:E ratio (1:x)", 0.5, 4),
        ("etco2", "EtCO2 (mmHg)", 10, 80),
    ]
    right_col = [
        ("etco2std", "EtCO2 std (mmHg)", 0, 5),
        ("risetime", "Phase II rise (s)", 0.05, 1.5),
        ("slope", "Phase III slope (mmHg/s)", 0, 10),
        ("fico2", "Baseline FiCO2 (mmHg)", 0, 15),
        ("anoise", "Noise (mmHg)", 0, 3),
    ]

    for row, (name, label, lo, hi) in enumerate(left_col):
        create_slider(name, label, [0.14, 0.38 - row * 0.065, 0.28, 0.03], lo, hi)
    for row, (name, label, lo, hi) in enumerate(right_col):
        create_slider(name, label, [0.64, 0.38 - row * 0.065, 0.28, 0.03], lo, hi)

    # ----------------------------------
    # Update
    # ----------------------------------

    state = {"busy": False}   # suspend le rafraîchissement pendant un réglage groupé des curseurs

    def redraw():
        try:
            t, co2, phases = generate()
            line.set_data(t, co2)
            for i in range(4):
                idx = np.where(phases == i)[0]
                markers[i].set_offsets(
                    np.column_stack((t[idx], co2[idx]))
                    if len(idx) > 0 else np.empty((0, 2))
                )
            ax.set_xlim(0, params["time_to_draw"])
            ax.set_ylim(0, max(50, np.max(co2) + 8))
            monitor.set_text(monitor_readings(t, co2, phases,
                                              params["time_to_draw"]))
            fig.canvas.draw_idle()
        except Exception as e:
            print("Capnogram generation error:", e)

    def update(value):
        if state["busy"]:
            return
        for name in sliders:
            params[name] = sliders[name].val
        redraw()

    for slider in sliders.values():
        slider.on_changed(update)

    def set_many(values):
        state["busy"] = True
        for name, v in values.items():
            sliders[name].set_val(v)
        state["busy"] = False
        update(None)

    # ----------------------------------
    # Presets
    # ----------------------------------

    fig.text(0.81, 0.70, "Clinical presets", fontsize=11, fontweight="bold")
    radio_ax = fig.add_axes([0.81, 0.50, 0.17, 0.19])
    radio = RadioButtons(radio_ax, list(PRESETS.keys()))
    radio.on_clicked(lambda label: set_many(PRESETS[label]))

    # ----------------------------------
    # Buttons
    # ----------------------------------

    reset_button = Button(fig.add_axes([0.81, 0.04, 0.07, 0.05]), "Reset")

    def reset(event):
        state["busy"] = True
        for s in sliders.values():
            s.reset()
        state["busy"] = False
        update(None)

    reset_button.on_clicked(reset)

    new_button = Button(fig.add_axes([0.89, 0.04, 0.09, 0.05]), "New breaths")

    def new_breaths(event):
        params["seed"] += 1
        redraw()

    new_button.on_clicked(new_breaths)

    redraw()
    plt.show()


if __name__ == "__main__":
    interactive_capno()
