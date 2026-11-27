import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from ecg import rrprocess   # même processus RR que l'ECG (variabilité de la fréquence cardiaque)

r'''
Implémentation de abpsyn —— courbe de pression artérielle invasive synthétique
(ABP, « arterial blood pressure », comme celle d'un cathéter artériel radial en réanimation)
Écrit sur le modèle de ecgsyn :

function [p, imarks] = abpsyn(sfabp, N, anoise, hrmean, hrstd, lfhfratio, sfint,
                              sys, dia, lvet, peakfrac, notch, tau, zc, regurg,
                              ppv, fresp, damping, fn, seed)

Sorties :
p:        pression artérielle (mmHg), un élément par échantillon
imarks:   tableau de même longueur que p, qui marque les points clés de chaque battement :
          0 = pied de l'onde (pression diastolique, début de l'éjection)
          1 = pic systolique
          2 = incisure dicrote (fermeture de la valve aortique)
          3 = onde dicrote (rebond après l'incisure)
          -1 = point ordinaire

Entrées :
sfabp:    fréquence d'échantillonnage de la sortie (Hz)
N:        nombre de battements cardiaques
anoise:   amplitude du bruit additif uniforme (mmHg)
hrmean:   fréquence cardiaque moyenne (battements/minute)
hrstd:    écart-type de la fréquence cardiaque
lfhfratio:rapport LF/HF de la variabilité cardiaque (voir rrprocess dans ecg.py)
sfint:    fréquence d'échantillonnage interne, multiple entier de sfabp
sys:      pression systolique visée (mmHg), normale ~120
dia:      pression diastolique visée (mmHg), normale ~80
lvet:     durée d'éjection ventriculaire à 60 bpm (s), ~0,33 ; elle varie avec sqrt(RR)
peakfrac: position du pic de débit dans l'éjection (0~1), ~0,3 en temps normal ;
          plus grande = montée lente (rétrécissement aortique : « pulsus tardus »)
notch:    profondeur de l'incisure dicrote (0 = absente)
tau:      constante de temps de la décroissance diastolique = R*C (s), ~1,5 ;
          plus petite = artères plus rigides ou vasodilatation
zc:       impédance caractéristique de l'aorte (rapportée à R), rend la montée plus raide
regurg:   fuite diastolique (0~1), simule une insuffisance aortique
ppv:      variation respiratoire de la pression pulsée (%), « Pulse Pressure Variation » ;
          > 13 % suggère une hypovolémie (patient répondeur au remplissage)
fresp:    fréquence respiratoire (Hz), 0,2 Hz = 12 cycles/minute
damping:  coefficient d'amortissement du capteur (ζ), optimal ~0,6~0,7 ;
          < 0,4 = ligne sous-amortie (systole surestimée), > 1 = ligne sur-amortie
fn:       fréquence propre du système cathéter + capteur (Hz)
seed:     graine aléatoire ; une fois fixée, la courbe ne saute pas quand on bouge les curseurs

Principe du modèle (modèle de Windkessel à 3 éléments) :

   débit cardiaque Q(t) ──> [ Zc ] ──┬──> pression P
                                     │
                                  [R]  [C]     C dPc/dt = Q - Pc/R
                                     │         P = Pc + Zc*Q
                                    ─┴─

   - pendant l'éjection (systole), le ventricule envoie un débit Q > 0 ;
   - à la fermeture de la valve aortique, un bref reflux crée l'incisure dicrote ;
   - pendant la diastole, Q = 0 et la pression décroît exponentiellement (constante tau).

Forme d'un battement :

  P |        1 pic systolique
    |       /\
    |      /  \  2 incisure
    |     /    \/\ 3 onde dicrote
    |    /        \___
    |   /             \____
    |__0 pied              \__0 (battement suivant)
    +------------------------------> t
       |<-systole->|<-- diastole -->|
'''


def beat_durations(N, hrmean, hrstd, lfhfratio):
    '''
    Durée (s) de chaque battement, à partir du processus RR de ecg.py.
    rrprocess donne une série RR échantillonnée à 1 Hz ; on lit la valeur
    au moment où commence chaque battement.
    '''
    rrmean = 60 / hrmean
    Nrr = 2 ** max(2, math.ceil(np.log2(N * rrmean * 1.5 + 2)))
    rr0 = rrprocess(0.1, 0.25, 0.01, 0.01, lfhfratio, hrmean, hrstd, 1, Nrr)
    T = np.empty(N)
    t = 0.0
    for k in range(N):
        T[k] = np.interp(t, np.arange(Nrr), rr0)
        t += T[k]
    return np.clip(T, 0.25, 3.0)


def local_min(x, i0, i1):
    '''Premier minimum local de x dans [i0, i1), ou None s'il n'y en a pas'''
    i0 = max(i0, 1)
    i1 = min(i1, len(x) - 1)
    for i in range(i0, i1):
        if x[i] <= x[i - 1] and x[i] < x[i + 1]:
            return i
    return None


def abpsyn(sfabp=250,
           N=10,
           anoise=0,
           hrmean=75,
           hrstd=1,
           lfhfratio=0.6,
           sfint=1000,
           sys=120,
           dia=80,
           lvet=0.33,
           peakfrac=0.3,
           notch=0.6,
           tau=1.5,
           zc=0.08,
           regurg=0.0,
           ppv=5,
           fresp=0.2,
           damping=0.65,
           fn=20,
           seed=None):

    # vérifier que sfint est un multiple entier de sfabp (comme dans ecgsyn)
    q = int(round(sfint / sfabp))
    if q * sfabp != sfint:
        print('sfint must be integer multiple of sfabp')
        return

    if seed is not None:
        np.random.seed(seed)    # rrprocess utilise np.random
    N = int(N)
    dia = min(dia, sys - 5)     # la diastolique doit rester sous la systolique

    # ---------- 1. Durée de chaque battement ----------
    T = beat_durations(N, hrmean, hrstd, lfhfratio)
    starts = np.concatenate(([0.0], np.cumsum(T)[:-1]))
    Ts = np.minimum(lvet * np.sqrt(T), 0.6 * T)       # durée de la systole

    # ---------- 2. Débit éjecté par le ventricule Q(t) ----------
    dt = 1 / sfint
    n = int(np.floor(np.sum(T) * sfint))
    t = np.arange(n) * dt
    Q = np.zeros(n)

    # exposant qui place le pic de sin(pi * x^a) en x = peakfrac
    a = np.log(0.5) / np.log(np.clip(peakfrac, 0.05, 0.95))

    for k in range(N):
        beat = (t >= starts[k]) & (t < starts[k] + T[k])
        tb = t[beat] - starts[k]
        # volume d'éjection modulé par la respiration (origine de la PPV)
        sv = 1 + (ppv / 200) * np.sin(2 * np.pi * fresp * starts[k])

        qk = np.zeros(len(tb))
        ej = tb < Ts[k]
        qk[ej] = np.sin(np.pi * (tb[ej] / Ts[k]) ** a)                     # éjection
        qk -= notch * 0.5 * np.exp(-0.5 * ((tb - Ts[k] - 0.015) / 0.012) ** 2)  # reflux bref -> incisure
        di = tb >= Ts[k]
        qk[di] -= regurg * 0.4 * np.exp(-(tb[di] - Ts[k]) / 0.25)          # fuite aortique
        Q[beat] = sv * qk

    # ---------- 3. Windkessel à 3 éléments ----------
    # C dPc/dt = Q - Pc/R, résolu exactement pas à pas par un filtre du 1er ordre :
    # Pc[i] = alpha * Pc[i-1] + R * (1 - alpha) * Q[i]
    R = 1.0
    alpha = np.exp(-dt / tau)
    b_wk, a_wk = [R * (1 - alpha)], [1, -alpha]
    # Valeur initiale en régime établi : on cherche Pc0 tel que le premier battement,
    # s'il se répétait, finisse à la même valeur qu'au départ (pas de dérive au début).
    # Système linéaire : Pc_fin = alpha^L * Pc0 + Pc_fin(départ à 0)  =>  Pc0 = Pc_fin(0) / (1 - alpha^L)
    L = int(T[0] * sfint)
    forced = signal.lfilter(b_wk, a_wk, Q[:L])[-1]
    Pc0 = forced / (1 - alpha ** L)
    Pc, _ = signal.lfilter(b_wk, a_wk, Q, zi=[alpha * Pc0])
    P = Pc + zc * Q

    # ---------- 4. Mise à l'échelle vers sys / dia (comme ecgsyn le fait en mV) ----------
    bidx = (starts * sfint).astype(int)
    bend = np.append(bidx[1:], n)
    pmax = np.median([P[i:j].max() for i, j in zip(bidx, bend)])
    pmin = np.median([P[i:j].min() for i, j in zip(bidx, bend)])
    P = dia + (P - pmin) * (sys - dia) / (pmax - pmin)

    # ---------- 5. Réponse du capteur (système masse-ressort du 2e ordre) ----------
    # H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    wn = 2 * np.pi * fn
    bz, az = signal.bilinear([wn ** 2], [1, 2 * damping * wn, wn ** 2], fs=sfint)
    Pm, _ = signal.lfilter(bz, az, P, zi=signal.lfilter_zi(bz, az) * P[0])

    # ---------- 6. Sous-échantillonnage vers sfabp ----------
    p = Pm[::q]
    m = len(p)

    # ---------- 7. Repérage des points clés ----------
    imarks = -np.ones(m, dtype=int)
    for k in range(N):
        s = int(starts[k] * sfabp)
        e = min(int((starts[k] + T[k]) * sfabp), m)
        ts = int(Ts[k] * sfabp)
        if e - s < 4:
            continue

        # 0 : pied = minimum autour du début du battement
        i0 = max(s - int(0.05 * sfabp), 0)
        i1 = min(s + int(0.10 * sfabp), m)
        foot = i0 + int(np.argmin(p[i0:i1]))
        imarks[foot] = 0

        # 1 : pic systolique = maximum pendant l'éjection
        i1 = min(s + ts + int(0.03 * sfabp), m)
        peak = s + int(np.argmax(p[s:i1]))
        imarks[peak] = 1

        # 2 et 3 : incisure (minimum local) puis onde dicrote (maximum qui suit)
        if notch > 0:
            nt = local_min(p, s + ts - int(0.03 * sfabp), s + ts + int(0.08 * sfabp))
            if nt is not None:
                imarks[nt] = 2
                j1 = min(nt + int(0.12 * sfabp), m)
                if j1 > nt + 1:
                    imarks[nt + int(np.argmax(p[nt:j1]))] = 3

    # ---------- 8. Bruit de mesure uniforme (comme dans ecgsyn) ----------
    eta = 2 * np.random.rand(m) - 1
    p = p + anoise * eta

    return (p, imarks)


MARK_COLORS = ["#2878B5", "#D62728", "#2CA02C", "#9467BD"]
MARK_LABELS = ["Diastolic (foot)", "Systolic peak", "Dicrotic notch", "Dicrotic wave"]


def monitor_readings(t, p, imarks, window):
    '''
    Comme un moniteur : SYS/DIA (PAM), fréquence cardiaque et PPV,
    calculées uniquement sur la partie visible de la courbe.
    '''
    feet = np.where((imarks == 0) & (t <= window))[0]
    peaks = np.where((imarks == 1) & (t <= window))[0]
    if len(feet) < 2 or len(peaks) < 1:
        return "ART   ---/---\n      (---)\nHR     -- /min\nPPV    -- %"

    sys_v = np.mean(p[peaks])
    dia_v = np.mean(p[feet])
    # PAM = moyenne de la pression sur des battements complets
    pam = np.mean(p[feet[0]:feet[-1]])
    hr = 60 / np.mean(np.diff(t[feet]))

    # pression pulsée de chaque battement : pic - pied qui le précède
    pp = []
    for pk in peaks:
        before = feet[feet < pk]
        if len(before) > 0:
            pp.append(p[pk] - p[before[-1]])
    if len(pp) >= 2:
        pp = np.array(pp)
        ppv = 100 * (pp.max() - pp.min()) / ((pp.max() + pp.min()) / 2)
        ppv_txt = f"{ppv:4.0f}"
    else:
        ppv_txt = "  --"

    return (f"ART  {sys_v:3.0f}/{dia_v:<3.0f}\n"
            f"      ({pam:3.0f})\n"
            f"HR   {hr:4.0f} /min\n"
            f"PPV  {ppv_txt} %")


def main():
    sfabp = 250
    p, marks = abpsyn(sfabp=sfabp, N=6, seed=0)
    t = np.arange(len(p)) / sfabp

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, p, color="#C0392B", linewidth=1.6, label="ABP")
    for i in range(4):
        idx = np.where(marks == i)[0]
        ax.scatter(t[idx], p[idx], color=MARK_COLORS[i], s=25,
                   zorder=5, label=MARK_LABELS[i])

    ax.set_title("Synthetic Arterial Blood Pressure", fontsize=17,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Pressure (mmHg)", fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(40, 160)
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.legend(loc="upper right", ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("synthetic_ABP.png", dpi=300, bbox_inches="tight")


def calc_N(hrmean, time_to_draw):
    # quelques battements de plus pour remplir la fenêtre même si le cœur ralentit
    return max(int(np.ceil(hrmean / 60 * time_to_draw * 1.3)) + 2, 3)


# Paramètres prédéfinis pour des situations cliniques courantes
PRESETS = {
    "Normal": dict(hrmean=75, sys=120, dia=80, peakfrac=0.3, notch=0.6,
                   tau=1.5, regurg=0.0, ppv=5, damping=0.65, fn=20),
    "Hypertension": dict(hrmean=80, sys=170, dia=100, peakfrac=0.3, notch=0.5,
                         tau=1.8, regurg=0.0, ppv=5, damping=0.65, fn=20),
    "Hypovolemic shock": dict(hrmean=120, sys=85, dia=55, peakfrac=0.3, notch=0.3,
                              tau=1.0, regurg=0.0, ppv=22, damping=0.65, fn=20),
    "Aortic stenosis": dict(hrmean=70, sys=105, dia=80, peakfrac=0.6, notch=0.15,
                            tau=1.5, regurg=0.0, ppv=5, damping=0.65, fn=20),
    "Aortic regurgitation": dict(hrmean=80, sys=160, dia=45, peakfrac=0.25, notch=0.1,
                                 tau=1.2, regurg=0.6, ppv=5, damping=0.65, fn=20),
    "Underdamped line": dict(hrmean=75, sys=120, dia=80, peakfrac=0.3, notch=0.6,
                             tau=1.5, regurg=0.0, ppv=5, damping=0.12, fn=8),
    "Overdamped line": dict(hrmean=75, sys=120, dia=80, peakfrac=0.3, notch=0.6,
                            tau=1.5, regurg=0.0, ppv=5, damping=1.5, fn=8),
}


def interactive_abp():

    from matplotlib.widgets import Slider, Button, RadioButtons

    # ----------------------------------
    # Initial parameters
    # ----------------------------------

    params = {
        "time_to_draw": 8,
        "sfabp": 250,
        "hrmean": 75,
        "hrstd": 1.0,
        "sys": 120,
        "dia": 80,
        "ppv": 5,
        "peakfrac": 0.3,
        "notch": 0.6,
        "tau": 1.5,
        "regurg": 0.0,
        "damping": 0.65,
        "fn": 20,
        "rr_resp": 12,
        "anoise": 0.0,
        "seed": 0,
    }

    def generate():
        p, marks = abpsyn(
            sfabp=params["sfabp"],
            N=calc_N(params["hrmean"], params["time_to_draw"]),
            anoise=params["anoise"],
            hrmean=params["hrmean"],
            hrstd=params["hrstd"],
            sys=params["sys"],
            dia=params["dia"],
            peakfrac=params["peakfrac"],
            notch=params["notch"],
            tau=params["tau"],
            regurg=params["regurg"],
            ppv=params["ppv"],
            damping=params["damping"],
            fn=params["fn"],
            fresp=params["rr_resp"] / 60,
            seed=params["seed"],
        )
        t = np.arange(len(p)) / params["sfabp"]
        return t, p, marks

    t, p, marks = generate()

    # ----------------------------------
    # Figure
    # ----------------------------------

    fig, ax = plt.subplots(figsize=(15, 8.5))
    plt.subplots_adjust(left=0.08, bottom=0.50, right=0.78, top=0.90)
    fig.suptitle("Interactive Arterial Blood Pressure Simulator",
                 fontsize=18, fontweight="bold")

    line, = ax.plot(t, p, color="#C0392B", linewidth=1.6)
    ax.set_title("Synthetic invasive arterial pressure (ABP)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.grid(color="#DDDDDD", linewidth=0.5)

    markers = []
    for i in range(4):
        idx = np.where(marks == i)[0]
        markers.append(ax.scatter(t[idx], p[idx], color=MARK_COLORS[i],
                                  s=25, zorder=5, label=MARK_LABELS[i]))
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    # Affichage façon moniteur (rouge = pression artérielle, comme en réanimation)
    monitor = fig.text(
        0.81, 0.73, "", fontsize=15, family="monospace",
        color="#FF4C4C", fontweight="bold",
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
        ("time_to_draw", "Window (s)", 2, 30),
        ("hrmean", "Heart rate (BPM)", 30, 180),
        ("hrstd", "HR std (BPM)", 0, 10),
        ("sys", "Systolic (mmHg)", 60, 220),
        ("dia", "Diastolic (mmHg)", 20, 130),
        ("ppv", "Resp. PPV (%)", 0, 40),
        ("rr_resp", "Resp. rate (/min)", 6, 40),
    ]
    right_col = [
        ("peakfrac", "Upstroke delay", 0.1, 0.8),
        ("notch", "Dicrotic notch depth", 0, 1.5),
        ("tau", "Diastolic decay RC (s)", 0.3, 4),
        ("regurg", "Aortic regurgitation", 0, 1),
        ("damping", "Line damping ζ", 0.05, 2),
        ("fn", "Line natural freq (Hz)", 4, 40),
        ("anoise", "Noise (mmHg)", 0, 5),
    ]

    for row, (name, label, lo, hi) in enumerate(left_col):
        create_slider(name, label, [0.14, 0.41 - row * 0.05, 0.28, 0.026], lo, hi)
    for row, (name, label, lo, hi) in enumerate(right_col):
        create_slider(name, label, [0.64, 0.41 - row * 0.05, 0.28, 0.026], lo, hi)

    # ----------------------------------
    # Update
    # ----------------------------------

    state = {"busy": False}   # suspend le rafraîchissement pendant un réglage groupé des curseurs

    def redraw():
        try:
            t, p, marks = generate()
            line.set_data(t, p)
            for i in range(4):
                idx = np.where(marks == i)[0]
                markers[i].set_offsets(
                    np.column_stack((t[idx], p[idx]))
                    if len(idx) > 0 else np.empty((0, 2))
                )
            ax.set_xlim(0, params["time_to_draw"])
            visible = p[t <= params["time_to_draw"]]
            ax.set_ylim(max(0, visible.min() - 15), visible.max() + 25)
            monitor.set_text(monitor_readings(t, p, marks,
                                              params["time_to_draw"]))
            fig.canvas.draw_idle()
        except Exception as e:
            print("ABP generation error:", e)

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

    fig.text(0.81, 0.66, "Clinical presets", fontsize=11, fontweight="bold")
    radio_ax = fig.add_axes([0.81, 0.45, 0.18, 0.20])
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

    new_button = Button(fig.add_axes([0.89, 0.04, 0.09, 0.05]), "New beats")

    def new_beats(event):
        params["seed"] += 1
        redraw()

    new_button.on_clicked(new_beats)

    redraw()
    plt.show()


if __name__ == "__main__":
    interactive_abp()
