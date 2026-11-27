import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from ecg import rrprocess   # même processus RR que l'ECG (variabilité de la fréquence cardiaque)

r'''
Implémentation de spo2syn —— oxymètre de pouls synthétique
(courbe pléthysmographique + calcul de la SpO2, comme un capteur au doigt)
Écrit sur le modèle de ecgsyn :

function [pleth, imarks, red, ir, spo2, sao2] = spo2syn(sfppg, N, anoise, hrmean, hrstd,
                                   lfhfratio, sao2, pi, reflect, pvi, fresp, cohb,
                                   motion, desat, desat_t, desat_dur, seed)

Principe de l'oxymètre de pouls :
  Le capteur envoie deux lumières à travers le doigt : rouge (660 nm) et infrarouge (940 nm).
  - L'hémoglobine oxygénée (HbO2) absorbe peu le rouge, la désoxygénée (Hb) l'absorbe beaucoup.
  - À chaque battement, le volume de sang artériel augmente : la lumière transmise baisse.
    C'est la partie pulsatile « AC », posée sur une partie continue « DC » (tissus, sang veineux).
  - L'appareil calcule le « rapport des rapports » :
        R = (AC_rouge / DC_rouge) / (AC_ir / DC_ir)
    puis le convertit en saturation par une courbe d'étalonnage empirique :
        SpO2 ≈ 110 - 25 * R        (R ≈ 0,5 -> ~98 %,  R ≈ 1 -> ~85 %)
  Ici on fait le chemin inverse pour générer les signaux, puis on refait le calcul
  comme un vrai oxymètre pour obtenir la SpO2 affichée.

Sorties :
pleth:    courbe pléthysmographique affichée (% du DC infrarouge), comme sur un moniteur
imarks:   tableau de même longueur que pleth, qui marque les points clés :
          0 = pied de l'onde (début du battement)
          1 = pic systolique
          2 = incisure dicrote
          3 = pic diastolique (onde réfléchie)
          -1 = point ordinaire
red, ir:  intensités lumineuses brutes reçues par le capteur (unités arbitraires)
spo2:     SpO2 affichée (%), moyennée sur 4 battements, NaN avant le 1er résultat
sao2:     saturation artérielle réelle au doigt (%), pour comparaison

Entrées :
sfppg:    fréquence d'échantillonnage (Hz)
N:        nombre de battements cardiaques
anoise:   bruit électronique uniforme (% du DC), gênant surtout si la perfusion est faible
hrmean:   fréquence cardiaque moyenne (battements/minute)
hrstd:    écart-type de la fréquence cardiaque
lfhfratio:rapport LF/HF (voir rrprocess dans ecg.py)
sao2:     saturation artérielle réelle (%), normale 95~100
pi:       indice de perfusion = AC/DC infrarouge (%), normal ~1~5 ;
          < 0,5 = mauvaise perfusion (froid, choc, vasoconstriction)
reflect:  amplitude relative de l'onde diastolique (réfléchie), plus grande si artères rigides
pvi:      variation respiratoire de l'amplitude du pléthysmogramme (%) ;
          élevée en cas d'hypovolémie (équivalent du PPV de la pression artérielle)
fresp:    fréquence respiratoire (Hz)
cohb:     carboxyhémoglobine (%) ; l'oxymètre la confond avec HbO2 -> SpO2 faussement normale
motion:   amplitude de l'artefact de mouvement (% du DC) ;
          il touche le rouge et l'infrarouge de la même façon -> R tend vers 1 -> SpO2 ~85 %
desat:    profondeur d'une désaturation transitoire (points de %), 0 = aucune
desat_t:  instant du creux de la désaturation (s)
desat_dur:durée de la désaturation (s)
seed:     graine aléatoire ; une fois fixée, la courbe ne saute pas quand on bouge les curseurs

Forme d'un battement (somme de deux gaussiennes, comme les ondes PQRST de ecgsyn) :

  pleth |      1 pic systolique
        |     /\
        |    /  \ 2 incisure
        |   /    \_  3 pic diastolique (souvent un simple épaulement)
        |  /       \
        |_0         \_____0 (battement suivant)
        +--------------------------> t
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


def spo2syn(sfppg=125,
            N=10,
            anoise=0,
            hrmean=75,
            hrstd=1,
            lfhfratio=0.6,
            sao2=98,
            pi=3.0,
            reflect=0.5,
            pvi=5,
            fresp=0.2,
            cohb=0,
            motion=0,
            desat=0,
            desat_t=40,
            desat_dur=40,
            seed=None):

    if seed is not None:
        np.random.seed(seed)    # rrprocess et le bruit utilisent np.random
    N = int(N)

    # ---------- 1. Durée de chaque battement ----------
    T = beat_durations(N, hrmean, hrstd, lfhfratio)
    starts = np.concatenate(([0.0], np.cumsum(T)[:-1]))

    n = int(np.floor(np.sum(T) * sfppg))
    t = np.arange(n) / sfppg

    # ---------- 2. Onde de pouls « propre » : deux gaussiennes par battement ----------
    # Les positions et largeurs varient avec sqrt(RR), comme la systole dans ecgsyn.
    # L'onde systolique est asymétrique : montée rapide (largeur 0,05), descente lente (0,10).
    def gauss(tb, mu, s_up, s_down):
        s = np.where(tb < mu, s_up, s_down)
        return np.exp(-0.5 * ((tb - mu) / s) ** 2)

    p = np.zeros(n)
    for k in range(N):
        sq = math.sqrt(T[k])
        # amplitude modulée par la respiration (origine du PVI)
        amp = 1 + (pvi / 200) * np.sin(2 * np.pi * fresp * starts[k])
        i0 = max(int((starts[k] - 0.5) * sfppg), 0)
        i1 = min(int((starts[k] + T[k] + 1.0) * sfppg), n)
        tb = t[i0:i1] - starts[k]
        p[i0:i1] += amp * (gauss(tb, 0.17 * sq, 0.05 * sq, 0.10 * sq)
                           + reflect * gauss(tb, 0.40 * sq, 0.12 * sq, 0.12 * sq))

    # Normalisation : pied médian -> 0, pic médian -> 1
    bidx = (starts * sfppg).astype(int)
    bend = np.append(bidx[1:], n)
    pmax = np.median([p[i:j].max() for i, j in zip(bidx, bend)])
    pmin = np.median([p[i:j].min() for i, j in zip(bidx, bend)])
    p = (p - pmin) / (pmax - pmin)

    # ---------- 3. Saturation réelle au doigt (avec désaturation éventuelle) ----------
    sat = np.full(n, float(sao2))
    if desat > 0:
        x = (t - desat_t) / desat_dur
        bump = np.where(np.abs(x) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * x)), 0)
        sat = sat - desat * bump
    sat = np.clip(sat, 0, 100)

    # Ce que « voit » l'oxymètre : COHb absorbe le rouge comme HbO2
    sat_seen = np.clip(sat + cohb, 0, 100)
    R = (110 - sat_seen) / 25                     # inverse de la courbe d'étalonnage

    # ---------- 4. Intensités lumineuses reçues (loi de Beer-Lambert simplifiée) ----------
    # Plus il y a de sang dans le doigt, moins la lumière passe : I = DC * (1 - AC/DC * p)
    DC_ir, DC_red = 1.0, 0.6
    ir = DC_ir * (1 - (pi / 100) * p)
    red = DC_red * (1 - R * (pi / 100) * p)

    # Légère oscillation de la ligne de base avec la respiration (sang veineux)
    wander = 1 + 0.003 * np.sin(2 * np.pi * fresp * t)
    ir *= wander
    red *= wander

    # Artefact de mouvement : bruit basse fréquence (0,3~4 Hz), identique sur les deux lumières
    if motion > 0:
        b, a = signal.butter(2, [0.3, 4], btype="bandpass", fs=sfppg)
        m = signal.filtfilt(b, a, np.random.randn(n))
        m /= np.std(m)
        ir *= 1 + (motion / 100) * m
        red *= 1 + (motion / 100) * m

    # Bruit électronique indépendant sur chaque canal
    ir += DC_ir * (anoise / 100) * (2 * np.random.rand(n) - 1)
    red += DC_red * (anoise / 100) * (2 * np.random.rand(n) - 1)

    # ---------- 5. Calcul de la SpO2, comme un vrai oxymètre ----------
    # Pour chaque battement : AC = max - min, DC = moyenne, puis R et SpO2.
    # (Un vrai appareil doit d'abord détecter les battements ; ici on utilise
    #  les débuts de battement connus pour simplifier.)
    spo2_beat = np.full(N, np.nan)
    for k in range(N):
        i, j = bidx[k], bend[k]
        if j - i < 3:
            continue
        ac_r, dc_r = np.ptp(red[i:j]), np.mean(red[i:j])
        ac_i, dc_i = np.ptp(ir[i:j]), np.mean(ir[i:j])
        Rk = (ac_r / dc_r) / (ac_i / dc_i)
        spo2_beat[k] = np.clip(110 - 25 * Rk, 0, 100)

    # Valeur affichée : moyenne des 4 derniers battements, mise à jour à la fin de chaque
    # battement (d'où un léger retard, comme sur un vrai moniteur)
    spo2 = np.full(n, np.nan)
    for k in range(N):
        val = np.nanmean(spo2_beat[max(0, k - 3):k + 1])
        spo2[bend[k] - 1:] = np.round(val)

    # ---------- 6. Courbe affichée et points clés ----------
    pleth = (1 - ir / DC_ir) * 100                # en % du DC, l'amplitude vaut donc le PI

    imarks = -np.ones(n, dtype=int)
    for k in range(N):
        s, e = bidx[k], bend[k]
        sq = math.sqrt(T[k])
        if e - s < 4:
            continue
        # 0 : pied = minimum juste avant la montée systolique
        i0 = max(s - int(0.05 * sfppg), 0)
        i1 = min(s + int(0.18 * sq * sfppg), n)
        if i1 > i0:
            imarks[i0 + int(np.argmin(p[i0:i1]))] = 0
        # 1 : pic systolique
        i1 = min(s + int(0.35 * sq * sfppg), n)
        peak = s + int(np.argmax(p[s:i1]))
        imarks[peak] = 1
        # 2 et 3 : incisure puis pic diastolique.
        # Si l'incisure forme un vrai creux, on prend le minimum local puis le maximum qui suit.
        # Sinon (cas fréquent : simple « épaulement »), on utilise la courbure :
        # incisure = courbure maximale vers le haut, pic diastolique = courbure maximale vers le bas.
        w1 = min(peak + int(0.4 * sq * sfppg), e)
        nt = local_min(p, peak + 1, w1)
        if nt is not None:
            imarks[nt] = 2
            j1 = min(nt + int(0.25 * sq * sfppg), n)
            if j1 > nt + 1:
                imarks[nt + int(np.argmax(p[nt:j1]))] = 3
        elif reflect > 0.05:
            j0 = peak + int(0.05 * sq * sfppg)
            if w1 - j0 > 4:
                d2 = np.gradient(np.gradient(p[j0:w1]))
                nt = j0 + int(np.argmax(d2))
                imarks[nt] = 2
                j1 = min(nt + int(0.2 * sq * sfppg), e)
                if j1 - nt > 2:
                    d2b = np.gradient(np.gradient(p[nt:j1]))
                    imarks[nt + int(np.argmin(d2b))] = 3

    return (pleth, imarks, red, ir, spo2, sat)


MARK_COLORS = ["#2878B5", "#D62728", "#2CA02C", "#9467BD"]
MARK_LABELS = ["Foot", "Systolic peak", "Dicrotic notch", "Diastolic peak"]


def monitor_readings(t, pleth, imarks, spo2, sao2, window):
    '''
    Comme un moniteur : SpO2, fréquence du pouls (PR) et indice de perfusion (PI),
    lus sur la partie visible de la courbe. Renvoie aussi la vraie SaO2 pour comparer.
    '''
    vis = t <= window
    shown = spo2[vis]
    shown = shown[~np.isnan(shown)]
    spo2_txt = f"{shown[-1]:3.0f}" if len(shown) else " --"
    sat_true = sao2[vis][-1] if np.any(vis) else np.nan

    feet = np.where((imarks == 0) & vis)[0]
    if len(feet) >= 2:
        pr_txt = f"{60 / np.mean(np.diff(t[feet])):4.0f}"
        pis = [np.ptp(pleth[a:b]) for a, b in zip(feet[:-1], feet[1:])]
        pi_txt = f"{np.mean(pis):4.1f}"
    else:
        pr_txt, pi_txt = "  --", "  --"

    text = (f"SpO2  {spo2_txt} %\n"
            f"PR   {pr_txt} /min\n"
            f"PI   {pi_txt} %")
    low = len(shown) > 0 and shown[-1] < 90
    return text, low, sat_true


def main():
    sfppg = 125
    pleth, marks, red, ir, spo2, sao2 = spo2syn(sfppg=sfppg, N=8, seed=0)
    t = np.arange(len(pleth)) / sfppg

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, pleth, color="#0097A7", linewidth=1.6, label="Pleth")
    for i in range(4):
        idx = np.where(marks == i)[0]
        ax.scatter(t[idx], pleth[idx], color=MARK_COLORS[i], s=25,
                   zorder=5, label=MARK_LABELS[i])

    ax.set_title("Synthetic Photoplethysmogram (pulse oximetry)", fontsize=17,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Pleth (% of DC)", fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.legend(loc="upper right", ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("synthetic_SpO2.png", dpi=300, bbox_inches="tight")


def calc_N(hrmean, time_to_draw):
    # quelques battements de plus pour remplir la fenêtre même si le cœur ralentit
    return max(int(np.ceil(hrmean / 60 * time_to_draw * 1.3)) + 2, 3)


# Paramètres prédéfinis pour des situations cliniques courantes
_BASE = dict(time_to_draw=10, hrmean=75, sao2=98, pi=3.0, pvi=5, reflect=0.5,
             cohb=0, motion=0, anoise=0, desat=0)
PRESETS = {
    "Normal": dict(_BASE),
    "Hypoxemia": dict(_BASE, hrmean=105, sao2=85, pi=2.0),
    "Low perfusion": dict(_BASE, hrmean=95, sao2=96, pi=0.3, anoise=0.03, reflect=0.2),
    "Motion artifact": dict(_BASE, motion=1.5),
    "CO poisoning": dict(_BASE, hrmean=100, sao2=75, cohb=22),
    "Apnea desaturation": dict(_BASE, time_to_draw=90, desat=20),
    "Hypovolemia": dict(_BASE, hrmean=115, sao2=97, pi=1.0, pvi=25, reflect=0.3),
}


def interactive_spo2():

    from matplotlib.widgets import Slider, Button, RadioButtons

    # ----------------------------------
    # Initial parameters
    # ----------------------------------

    params = {
        "time_to_draw": 10,
        "sfppg": 125,
        "hrmean": 75,
        "hrstd": 1.0,
        "sao2": 98,
        "pi": 3.0,
        "pvi": 5,
        "rr_resp": 12,
        "reflect": 0.5,
        "cohb": 0,
        "motion": 0.0,
        "anoise": 0.0,
        "desat": 0,
        "desat_t": 40,
        "desat_dur": 40,
        "seed": 0,
    }

    def generate():
        pleth, marks, red, ir, spo2, sao2 = spo2syn(
            sfppg=params["sfppg"],
            N=calc_N(params["hrmean"], max(params["time_to_draw"],
                                           params["desat_t"] + params["desat_dur"] / 2
                                           if params["desat"] > 0 else 0)),
            anoise=params["anoise"],
            hrmean=params["hrmean"],
            hrstd=params["hrstd"],
            sao2=params["sao2"],
            pi=params["pi"],
            reflect=params["reflect"],
            pvi=params["pvi"],
            fresp=params["rr_resp"] / 60,
            cohb=params["cohb"],
            motion=params["motion"],
            desat=params["desat"],
            desat_t=params["desat_t"],
            desat_dur=params["desat_dur"],
            seed=params["seed"],
        )
        t = np.arange(len(pleth)) / params["sfppg"]
        return t, pleth, marks, spo2, sao2

    t, pleth, marks, spo2, sao2 = generate()

    # ----------------------------------
    # Figure : pléthysmogramme en haut, tendance SpO2 en dessous
    # ----------------------------------

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Interactive Pulse Oximetry (SpO2) Simulator",
                 fontsize=18, fontweight="bold")
    ax = fig.add_axes([0.07, 0.71, 0.70, 0.20])
    ax2 = fig.add_axes([0.07, 0.505, 0.70, 0.15], sharex=ax)

    line, = ax.plot(t, pleth, color="#0097A7", linewidth=1.5)
    ax.set_title("Plethysmogram (infrared, inverted)")
    ax.set_ylabel("Pleth (% of DC)")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.tick_params(labelbottom=False)

    markers = []
    for i in range(4):
        idx = np.where(marks == i)[0]
        markers.append(ax.scatter(t[idx], pleth[idx], color=MARK_COLORS[i],
                                  s=20, zorder=5, label=MARK_LABELS[i]))
    ax.legend(loc="upper right", ncol=4, fontsize=8)

    spo2_line, = ax2.step(t, spo2, where="post", color="#0097A7",
                          linewidth=1.8, label="SpO2 displayed")
    sao2_line, = ax2.plot(t, sao2, "--", color="#555555", linewidth=1.2,
                          label="True SaO2 (finger)")
    ax2.axhline(90, color="#D62728", linestyle=":", linewidth=1, label="Alarm 90 %")
    ax2.set_ylabel("Saturation (%)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(50, 102)
    ax2.grid(color="#DDDDDD", linewidth=0.5)
    ax2.legend(loc="lower left", ncol=3, fontsize=8)

    # Affichage façon moniteur (cyan = SpO2, rouge si alarme)
    monitor = fig.text(
        0.81, 0.76, "", fontsize=15, family="monospace",
        fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="#555555",
                  boxstyle="round,pad=0.6"),
    )
    true_txt = fig.text(0.81, 0.715, "", fontsize=10, color="#555555")

    # ----------------------------------
    # Sliders
    # ----------------------------------

    sliders = {}

    def create_slider(name, label, position, minimum, maximum):
        slider_ax = fig.add_axes(position)
        sliders[name] = Slider(slider_ax, label, minimum, maximum,
                               valinit=params[name])

    left_col = [
        ("time_to_draw", "Window (s)", 3, 120),
        ("hrmean", "Heart rate (BPM)", 30, 180),
        ("hrstd", "HR std (BPM)", 0, 10),
        ("sao2", "True SaO2 (%)", 50, 100),
        ("pi", "Perfusion index (%)", 0.1, 10),
        ("pvi", "Resp. variation PVI (%)", 0, 40),
        ("rr_resp", "Resp. rate (/min)", 6, 40),
    ]
    right_col = [
        ("reflect", "Diastolic wave ratio", 0, 1),
        ("cohb", "COHb (%)", 0, 40),
        ("motion", "Motion artifact (%DC)", 0, 3),
        ("anoise", "Noise (%DC)", 0, 0.1),
        ("desat", "Desaturation depth (%)", 0, 40),
        ("desat_t", "Desaturation center (s)", 0, 120),
        ("desat_dur", "Desaturation length (s)", 10, 120),
    ]

    for row, (name, label, lo, hi) in enumerate(left_col):
        create_slider(name, label, [0.15, 0.41 - row * 0.05, 0.27, 0.026], lo, hi)
    for row, (name, label, lo, hi) in enumerate(right_col):
        create_slider(name, label, [0.65, 0.41 - row * 0.05, 0.27, 0.026], lo, hi)

    # ----------------------------------
    # Update
    # ----------------------------------

    state = {"busy": False}   # suspend le rafraîchissement pendant un réglage groupé des curseurs

    def redraw():
        try:
            t, pleth, marks, spo2, sao2 = generate()
            line.set_data(t, pleth)
            for i in range(4):
                idx = np.where(marks == i)[0]
                markers[i].set_offsets(
                    np.column_stack((t[idx], pleth[idx]))
                    if len(idx) > 0 else np.empty((0, 2))
                )
            spo2_line.set_data(t, spo2)
            sao2_line.set_data(t, sao2)

            w = params["time_to_draw"]
            ax.set_xlim(0, w)
            visible = pleth[t <= w]
            margin = 0.15 * (visible.max() - visible.min() + 1e-6)
            ax.set_ylim(visible.min() - margin, visible.max() + 2.5 * margin)

            text, low, sat_true = monitor_readings(t, pleth, marks, spo2, sao2, w)
            monitor.set_text(text)
            monitor.set_color("#FF4C4C" if low else "#00E5FF")
            true_txt.set_text(f"(true SaO2 at {w:.0f} s: {sat_true:.0f} %)")
            fig.canvas.draw_idle()
        except Exception as e:
            print("SpO2 generation error:", e)

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

    fig.text(0.81, 0.665, "Clinical presets", fontsize=11, fontweight="bold")
    radio_ax = fig.add_axes([0.81, 0.45, 0.18, 0.21])
    radio = RadioButtons(radio_ax, list(PRESETS.keys()))
    radio.on_clicked(lambda label: set_many(PRESETS[label]))

    # ----------------------------------
    # Buttons
    # ----------------------------------

    reset_button = Button(fig.add_axes([0.81, 0.03, 0.07, 0.045]), "Reset")

    def reset(event):
        state["busy"] = True
        for s in sliders.values():
            s.reset()
        state["busy"] = False
        update(None)

    reset_button.on_clicked(reset)

    new_button = Button(fig.add_axes([0.89, 0.03, 0.09, 0.045]), "New beats")

    def new_beats(event):
        params["seed"] += 1
        redraw()

    new_button.on_clicked(new_beats)

    redraw()
    plt.show()


if __name__ == "__main__":
    interactive_spo2()
