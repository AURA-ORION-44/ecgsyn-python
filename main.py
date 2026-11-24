
import math
import numpy as np
from scipy.fft import ifft
import scipy.integrate as integrate

J = 1j

def ecgsyn(sfecg=256,
           N=256,
           anoise=0,
           hrmean=60,
           hrstd=1,
           lfhfratio=0.6,
           sfint=512,
           ti=[-70, -15, 0, 15, 100],
           ai=[1.2, -5, 30, -7.5, 0.75], # PQRST
           bi=[0.25, 0.1, 0.1, 0.1, 0.4] # PQRST
           ):

    # rr: rr process
    # sfint: Internal sampling frequency[Hertz]
    # Order of extrema: [P Q R S T]
    # ti = angles of extrema[radians]
    # ai = z - position of extrema
    # bi = Gaussian width of peaks
    
    # adjust extrema parameters for mean heart rate
    hrfact = math.sqrt(hrmean / 60)
    hrfact2 = math.sqrt(hrfact)
    bi = [b*hrfact for b in bi]

    ti = [tii*np.pi/180 for tii in ti]
    # Convert angular positions to radians

    # Preserve original ECGSYN scaling for PQRST,
    # while supporting additional waves.
    if len(ti) >= 5:

        ti = ti.copy()

        ti[0] *= hrfact2
        ti[1] *= hrfact
        ti[3] *= hrfact
        ti[4] *= hrfact2



    # check that sfint is an integer multiple of sfecg
    q = np.round(sfint/sfecg)
    qd = sfint//sfecg
    if not q == qd:
        print('sfint must be integer multiple of sfecg')
        return

    # define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and ressp.PIratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # calculate time scales for rr and total output
    sampfreqrr = 1
    trr = 1 / sampfreqrr
    tstep = 1 / sfecg
    rrmean = (60 / hrmean)
    Nrr = 2 ** (math.ceil(np.log2(N * rrmean / trr)))

    # compute rr process
    rr0 = rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sampfreqrr, Nrr)

    # upsample rr time series from 1 Hz to sfint Hz
    idx = np.arange(0.0, len(rr0), 1.0)
    idx_interp = np.arange(0.0, len(rr0), 1/sfint)
    rr = np.interp(idx_interp, idx, rr0)

    # make the rr n time series
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg = 0

    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(np.round(tecg / dt))-1
        for ii in np.arange(i, ip):
            if ii >= len(rr):
                break
            rrn[ii] = rr[i]
        i = ip
    Nt = len(rr)-1

    x0 = [1, 0, 0.04]
    Tspan = np.arange(0, (Nt)*dt, dt)

    '''ode with dop853'''
    # solver = integrate.ode(desrivecgsyn)
    # solver.set_integrator('dop853')
    # solver.set_f_params([], rrn, sfint, ti, ai, bi)
    # t0 = 0.0
    # solver.set_initial_value(x0, t0)
    # X0 = np.empty((len(Tspan), 3))
    # X0[0] = x0
    # k = 1
    # while solver.successful() and solver.t < (Nt-1)*dt:
    #     solver.integrate(Tspan[k])
    #     X0[k] = solver.y
    #     k += 1

    X0 = integrate.odeint(desrivecgsyn, x0, Tspan, args=([], rrn, sfint, ti, ai, bi))

    # downsample to required sfecg
    X = X0[::int(q)]

    # extract R - peaks times

    ipeaks = detectpeaks(X, ti, sfecg, ai)

    # Scale signal to line between - 0.4 and 1.2 mV
    z = [xx[2] for xx in X]
    zmin = np.min(z)
    zmax = np.max(z)
    zrange = zmax - zmin
    z = (z - zmin) * (1.6) / zrange - 0.4

    # include additive uniformly distributed measurement noise
    eta = 2 * np.random.rand(len(z)) - 1
    s = z + anoise * eta

    return (s, ipeaks)

def rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    w1 = 2*np.pi*flo
    w2 = 2*np.pi*fhi
    c1 = 2*np.pi*flostd
    c2 = 2*np.pi*fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60/hrmean
    rrstd = 60*hrstd/(hrmean*hrmean)

    df = sfrr/n
    w = np.arange(0, n)*2*np.pi*df
    dw1 = w-w1
    dw2 = w-w2

    Hw1 = sig1*np.exp(-0.5*(dw1/c1)**2)/np.sqrt(2*np.pi*c1**2)
    Hw2 = sig2*np.exp(-0.5*(dw2/c2)**2)/np.sqrt(2*np.pi*c2**2)
    Hw = Hw1 + Hw2
    Hw0 = []
    for h in Hw[0:n//2]:
        Hw0.append(h)
    for h in Hw[n//2-1::-1]:
        Hw0.append(h)
    Sw = [sfrr/2*np.sqrt(h) for h in Hw0]

    ph0 = 2*np.pi*np.random.rand(n//2-1)
    ph = []
    ph.append(0)
    for v in ph0:
        ph.append(v)
    ph.append(0)
    for v in ph0[::-1]:
        ph.append(-v)
    SwC = [Sw[m]*np.exp(J*ph[m]) for m in range(len(Sw))]
    x = (1/n)*np.real(ifft(SwC))

    xstd = np.std(x)

    ratio = rrstd/xstd

    rr = rrmean + x*ratio

    return rr

def detectpeaks(X, thetap, sfecg, ai):
    N = len(X)
    n_peaks = len(thetap)

    detected = -1 * np.ones(N)

    theta = np.array([
        np.arctan2(X[m][1], X[m][0])
        for m in range(N)
    ])

    ind0 = -1 * np.ones(N)

    # Associate angular positions with waveform peaks
    for i in range(N - 1):
        for j in range(n_peaks):

            target = thetap[j]

            if theta[i] <= target <= theta[i + 1]:
                if abs(target - theta[i]) < abs(theta[i + 1] - target):
                    ind0[i] = j
                else:
                    ind0[i + 1] = j

    d = max(2, int(np.ceil(sfecg / 64)))

    z = np.array([xx[2] for xx in X])

    # Find the local extremum corresponding to each wave
    for i in range(n_peaks):

        indices = np.where(ind0 == i)[0]

        for idx in indices:

            start = max(0, idx - d)
            end = min(N, idx + d + 1)

            segment = z[start:end]

            # Positive amplitude -> maximum
            # Negative amplitude -> minimum
            if ai[i] >= 0:
                peak_idx = start + np.argmax(segment)
            else:
                peak_idx = start + np.argmin(segment)

            detected[peak_idx] = i

    return detected

def desrivecgsyn(y0, t, flag, rr, sfint, ti, ai, bi):
# def desrivecgsyn(t, y0, flag, rr, sfint, ti, ai, bi):
    # function dxdt = derivsecgsyn(t, x, flag, rr, sfint, ti, ai, bi)
    # dxdt = derivsecgsyn(t, x, flag, rr, sampfreq, ti, ai, bi)
    # ODE file for generating the synthetic ECG
    # This file provides dxdt = F(t, x) taking input paramters:
    # rr: rr process
    # sfint: Internal sampling frequency[Hertz]
    # Order of extrema: [P Q R S T]
    # ti = angles of extrema[radians]
    # ai = z - position of extrema
    # bi = Gaussian width of peaks

    # Copyright(c) 2003 by Patrick McSharry & Gari Clifford, All Rights Reserved
    # See IEEE Transactions On Biomedical Engineering, 50(3), 289 - 294, March 2003.
    # Contact P.McSharry(patrick AT mcsharry DOT net) or % G.D.Clifford(gari AT mit DOT edu)

    xi = np.cos(ti)
    yi = np.sin(ti)
    ta = np.arctan2(y0[1], y0[0])
    r0 = 1
    a0 = 1.0 - np.sqrt(y0[0] ** 2 + y0[1]**2 ) / r0
    ip = int(np.floor(t * sfint))
    ip = np.clip(ip, 0, len(rr)-1)
    w0 = 2 * np.pi / rr[ip]

    w0 = 2 * np.pi / rr[ip]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * y0[0] - w0 * y0[1]
    dx2dt = a0 * y0[1] + w0 * y0[0]

    # dti = np.remainder(ta - ti, 2 * np.pi)
    #dti = [ta-tii for tii in ti]
    dti = np.arctan2(
    np.sin(ta - np.asarray(ti)),
    np.cos(ta - np.asarray(ti)))
    for m in range(len(dti)):
        if dti[m] >= 0:
            dti[m] = np.remainder(dti[m], 2*np.pi)
        else:
            dti[m] = -np.remainder(-dti[m],2*np.pi)

    dx3dt = -np.sum([ai[m]*dti[m]*np.exp(-0.5*(dti[m]/bi[m])**2) for m in range(len(ai))]) - 1.0*(y0[2]-zbase)

    return [dx1dt, dx2dt, dx3dt]


def calc_N(hrmean,time_to_draw):
    return max(np.ceil(hrmean/60*time_to_draw),3)


def interactive_ecg():

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import numpy as np

    # ==========================================
    # CONFIGURATION
    # ==========================================

    MAX_PEAKS = 8

    params = {
        "time_to_draw": 5,
        "anoise": 0.0,
        "hrmean": 100,
        "hrstd": 1.0,
        "lfhfratio": 0.6,
        "sfecg": 256,
        "sfint": 512
    }

    # Each wave is independently configurable
    ti = [-70, -15, 0, 15, 100]
    ai = [1.2, -5, 30, -7.5, 0.75]
    bi = [0.25, 0.1, 0.1, 0.1, 0.4]

    peak_colors = [
        "#2878B5",
        "#E69F00",
        "#D62728",
        "#2CA02C",
        "#9467BD",
        "#17BECF",
        "#8C564B",
        "#E377C2"
    ]

    # ==========================================
    # ECG GENERATION
    # ==========================================

    def generate():

        N = calc_N(
            params["hrmean"],
            params["time_to_draw"]
        )

        s, peaks = ecgsyn(
            sfecg=params["sfecg"],
            N=N,
            anoise=params["anoise"],
            hrmean=params["hrmean"],
            hrstd=params["hrstd"],
            lfhfratio=params["lfhfratio"],
            sfint=params["sfint"],
            ti=ti.copy(),
            ai=ai.copy(),
            bi=bi.copy()
        )

        t = np.arange(len(s)) / params["sfecg"]

        return t, s, peaks

    t, s, peaks = generate()

    # ==========================================
    # FIGURE
    # ==========================================

    fig = plt.figure(figsize=(16, 11))

    fig.suptitle(
        "Interactive ECGSYN Waveform Editor",
        fontsize=19,
        fontweight="bold",
        y=0.98
    )

    # ==========================================
    # ECG PLOT
    # ==========================================

    ax = fig.add_axes([0.09, 0.57, 0.85, 0.34])

    names = ["P","Q","R","S","T","U","V","W","X","Y","Z"]

    line, = ax.plot(
        t,
        s,
        color="#B22222",
        linewidth=1.3,
        label="ECG"
    )

    markers = []

    for i in range(MAX_PEAKS):

        marker = ax.scatter(
            [],
            [],
            color=peak_colors[i],
            s=35,
            label=names[i],
            zorder=5
        )

        markers.append(marker)

    ax.set_title("Synthetic ECG", fontsize=14)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")

    ax.set_xlim(0, params["time_to_draw"])

    ax.set_ylim(
        min(-1, np.min(s) - 0.2),
        max(2, np.max(s) + 0.2)
    )

    # ECG paper grid
    ax.set_xticks(np.arange(0, params["time_to_draw"], 0.2))
    ax.set_xticks(np.arange(0, params["time_to_draw"], 0.04), minor=True)

    ax.set_yticks(np.arange(-2, 3, 0.2))
    ax.set_yticks(np.arange(-2, 3, 0.04), minor=True)

    ax.grid(which="major", color="#E6A0A0", linewidth=0.5)
    ax.grid(which="minor", color="#F5DADA", linewidth=0.2)

    ax.legend(loc="upper right", ncol=4, fontsize=8)

    # ==========================================
    # GLOBAL PARAMETERS
    # ==========================================

    global_sliders = {}

    global_config = [
        ("time_to_draw", "Duration (s)", 0.1, 10),
        ("hrmean", "Heart rate (BPM)", 40, 180),
        ("hrstd", "HR variability", 0, 10),
        ("lfhfratio", "LF/HF ratio", 0.1, 3),
        ("anoise", "Noise amplitude", 0, 0.5)
    ]

    for i, (name, label, vmin, vmax) in enumerate(global_config):

        slider_ax = fig.add_axes([
            0.17,
            0.49 - i * 0.035,
            0.68,
            0.018
        ])

        slider = Slider(
            slider_ax,
            label,
            vmin,
            vmax,
            valinit=params[name]
        )

        global_sliders[name] = slider

    # ==========================================
    # PEAK EDITOR
    # ==========================================

    fig.text(
        0.09,
        0.305,
        "WAVEFORM EDITOR",
        fontsize=13,
        fontweight="bold"
    )

    fig.text(0.16, 0.275, "Position (ti)", fontsize=10)
    fig.text(0.46, 0.275, "Amplitude (ai)", fontsize=10)
    fig.text(0.76, 0.275, "Width (bi)", fontsize=10)


    def release_mouse_if_needed(fig):
        canvas = fig.canvas

        if canvas.mouse_grabber is not None:
            canvas.release_mouse(canvas.mouse_grabber)


    peak_sliders = []

    def create_peak_slider(i):

        y = 0.245 - i * 0.020

        # Position
        ax_ti = fig.add_axes([0.12, y, 0.23, 0.018])

        

        slider_ti = Slider(
            ax_ti,
            names[i],
            -180,
            180,
            valinit=ti[i] if i < len(ti) else 0
        )

        # Amplitude
        ax_ai = fig.add_axes([0.42, y, 0.23, 0.018])

        slider_ai = Slider(
            ax_ai,
            "",
            -30,
            35,
            valinit=ai[i] if i < len(ai) else 1
        )

        # Width
        ax_bi = fig.add_axes([0.72, y, 0.23, 0.018])

        slider_bi = Slider(
            ax_bi,
            "",
            0.02,
            0.5,
            valinit=bi[i] if i < len(bi) else 0.2
        )

        peak_sliders.append({
            "ti": slider_ti,
            "ai": slider_ai,
            "bi": slider_bi
        })

        return [ax_ti, ax_ai, ax_bi]

    peak_axes = []

    for i in range(MAX_PEAKS):

        axes = create_peak_slider(i)

        # Hide unused waves
        if i >= len(ti):
            for a in axes:
                a.set_visible(False)

        peak_axes.append(axes)

    # ==========================================
    # UPDATE FUNCTION
    # ==========================================

    updating = False

    def update(event=None):

        nonlocal updating

        if updating:
            return

        updating = True

        try:

            # Global parameters
            for name, slider in global_sliders.items():
                params[name] = slider.val

            # Wave parameters
            for i in range(len(ti)):

                ti[i] = peak_sliders[i]["ti"].val
                ai[i] = peak_sliders[i]["ai"].val
                bi[i] = peak_sliders[i]["bi"].val

            # Generate new ECG
            t_new, s_new, peaks_new = generate()

            line.set_data(t_new, s_new)

            # Update peak markers
            for i in range(MAX_PEAKS):

                if i < len(ti):

                    indices = np.where(peaks_new == i)[0]

                    offsets = np.column_stack((
                        t_new[indices],
                        s_new[indices]
                    ))

                    markers[i].set_offsets(offsets)

                else:

                    markers[i].set_offsets(
                        np.empty((0, 2))
                    )

            ax.set_xlim(
                0,
                params["time_to_draw"]
            )

            ax.set_ylim(
                min(-1, np.min(s_new) - 0.2),
                max(2, np.max(s_new) + 0.2)
            )

            fig.canvas.draw_idle()

        except Exception as e:

            print("ECG generation error:", e)

        finally:
            updating = False

    # ==========================================
    # CONNECT SLIDERS
    # ==========================================

    for slider in global_sliders.values():
        slider.on_changed(update)

    for group in peak_sliders:

        for slider in group.values():
            slider.on_changed(update)

    # ==========================================
    # ADD PEAK BUTTON
    # ==========================================

    add_ax = fig.add_axes([0.09, 0.010, 0.13, 0.04])

    add_button = Button(
        add_ax,
        "+ Add wave",
        color="#D9EAD3",
        hovercolor="#B6D7A8"
    )

    def add_peak(event):

        release_mouse_if_needed(fig)


        if len(ti) >= MAX_PEAKS:
            print("Maximum number of waves reached.")
            return

        i = len(ti)

        # Default parameters for a new wave
        new_ti = 140 if i == 5 else -120 if i == 6 else 60
        new_ai = 2.0
        new_bi = 0.2

        ti.append(new_ti)
        ai.append(new_ai)
        bi.append(new_bi)

        group = peak_sliders[i]

        for slider in group.values():
            slider.ax.set_visible(True)
            slider.set_active(True)

        update()

        fig.canvas.draw_idle()

    add_button.on_clicked(add_peak)

    # ==========================================
    # REMOVE PEAK BUTTON
    # ==========================================

    remove_ax = fig.add_axes([0.24, 0.010, 0.13, 0.04])

    remove_button = Button(
        remove_ax,
        "- Remove wave",
        color="#F4CCCC",
        hovercolor="#EA9999"
    )

    def remove_peak(event):

        release_mouse_if_needed(fig)


        if len(ti) <= 1:
            print("At least one wave must remain.")
            return

        i = len(ti) - 1

        ti.pop()
        ai.pop()
        bi.pop()

        group = peak_sliders[i]

        for slider in group.values():
            slider.set_active(False)
            slider.ax.set_visible(False)


        markers[i].set_offsets(
            np.empty((0, 2))
        )

        update()

        fig.canvas.draw_idle()

    remove_button.on_clicked(remove_peak)

    # ==========================================
    # RESET BUTTON
    # ==========================================

    reset_ax = fig.add_axes([0.84, 0.010, 0.11, 0.04])

    reset_button = Button(
        reset_ax,
        "Reset",
        color="#D9EAF7"
    )

    

    def reset(event):

        release_mouse_if_needed(fig)

        global_defaults = {
            "time_to_draw": 5,
            "hrmean": 100,
            "hrstd": 1,
            "lfhfratio": 0.6,
            "anoise": 0
        }

        defaults_ti = [-70, -15, 0, 15, 100]
        defaults_ai = [1.2, -5, 30, -7.5, 0.75]
        defaults_bi = [0.25, 0.1, 0.1, 0.1, 0.4]

        # Temporarily disable update callbacks
        for slider in global_sliders.values():
            slider.eventson = False

        for group in peak_sliders:
            for slider in group.values():
                slider.eventson = False

        # Restore global parameters
        for name, value in global_defaults.items():
            global_sliders[name].set_val(value)
            params[name] = value

        # Restore PQRST
        ti[:] = defaults_ti
        ai[:] = defaults_ai
        bi[:] = defaults_bi

        # Restore slider values and visibility
        for i, group in enumerate(peak_sliders):

            active = i < 5

            for slider in group.values():
                slider.ax.set_visible(active)

            if active:
                group["ti"].set_val(defaults_ti[i])
                group["ai"].set_val(defaults_ai[i])
                group["bi"].set_val(defaults_bi[i])

        # Re-enable callbacks
        for slider in global_sliders.values():
            slider.eventson = True

        for group in peak_sliders:
            for slider in group.values():
                slider.eventson = True

        update()

        fig.canvas.draw_idle()

    reset_button.on_clicked(reset)

    reset(0)


    # ==========================================
    # DISPLAY
    # ==========================================

    plt.show()

interactive_ecg()