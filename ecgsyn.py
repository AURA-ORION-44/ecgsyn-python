
import matplotlib.pyplot as plt

import math
import numpy as np
from scipy.fft import ifft
import scipy.integrate as integrate


J = 1j

'''
ecgsyn implementation
function [s, ipeaks] = ecgsyn(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,ti,ai,bi)
sfecg: sampling freq ecg
N: heart beat number
anoise: additive noise
hrmean
hrstd
lfhfratio: LF/HF ratio
sfint: internal sampling frequency
ti: angles of extremea [-70 -15 0 15 100] degrees
ai = z-position of extrema [1.2 -5 30 -7.5 0.75]
bi = Gaussian width of peaks [0.25 0.1 0.1 0.1 0.4]

====================================================================
Original copyrights:

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

ecgsyn.m and its dependents are freely availble from Physionet - 
http://www.physionet.org/ - please report any bugs to the authors above.
====================================================================
'''
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
    ti = [hrfact2*ti[0], hrfact*ti[1], ti[2], hrfact*ti[3], hrfact2*ti[4]]

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

    ipeaks = detectpeaks(X, ti, sfecg)

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


def detectpeaks(X, thetap, sfecg):
    N = len(X)
    irpeaks = np.zeros(N)

    theta = [np.arctan2(X[m][1], X[m][0]) for m in range(len(X))]
    ind0 = -1*np.ones(N)
    for i in range(N-1):
        j = [v for v in filter(lambda x:theta[i]<= thetap[x] and thetap[x] <= theta[i+1], range(len(thetap)))]
        if len(j) > 0:
            d1 = thetap[j[0]] - theta[i]
            d2 = theta[i + 1] - thetap[j[0]]

            if d1 < d2:
                ind0[i] = j[0]
            else:
                ind0[i+1] = j[0]

    d = np.ceil(sfecg / 64)
    d = np.max([2, d])
    ind = -1*np.ones(N)
    z = [xx[2] for xx in X]
    zmin = np.min(z)
    zmax = np.max(z)
    zext = [zmin, zmax, zmin, zmax, zmin]
    sext = [1, -1, 1, -1, 1]
    for i in range(5):
        # clear ind1 Z k vmax imax iext
        ind1 = [v for v in filter(lambda a: ind0[a]==i, range(len(ind0)))]

        if len(ind1) == 0:
            continue
        n = len(ind1)

        Z = np.ones(shape=(n, int(2 * d + 1))) * zext[i] * sext[i]
        for j in np.arange(-d,d+1,1):
            k = [v for v in filter(lambda a: 0<= ind1[a]+j and ind1[a]+j <= N-1, range(len(ind1)))]
            for kk in k:
                Z[kk][int(d + j)] = z[int(ind1[kk] + j-1)] * sext[i]

        vmax = np.max(Z, axis=1)
        ivmax = np.argmax(Z, axis=1)
        iext = np.add(ivmax,ind1)
        iext = np.add(iext, -d-2)
        for ii in iext:
            ind[int(ii)] = i

    return ind


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
    w0 = 2 * np.pi / rr[ip]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * y0[0] - w0 * y0[1]
    dx2dt = a0 * y0[1] + w0 * y0[0]

    # dti = np.remainder(ta - ti, 2 * np.pi)
    dti = [ta-tii for tii in ti]
    for m in range(len(dti)):
        if dti[m] >= 0:
            dti[m] = np.remainder(dti[m], 2*np.pi)
        else:
            dti[m] = -np.remainder(-dti[m],2*np.pi)

    dx3dt = -np.sum([ai[m]*dti[m]*np.exp(-0.5*(dti[m]/bi[m])**2) for m in range(len(ai))]) - 1.0*(y0[2]-zbase)

    return [dx1dt, dx2dt, dx3dt]

def main():
    sfecg=256
    N=3
    anoise=0
    hrmean=60
    hrstd=1
    lfhfratio=0.6
    sfint=512
    ti=[-70, -15, 0, 15, 100]
    ai=[1.2, -5, 30, -7.5, 0.75] # PQRST
    bi=[0.25, 0.1, 0.1, 0.1, 0.4] # PQRST

    s, peaks = ecgsyn(sfecg, N, anoise, hrmean, hrstd, lfhfratio, sfint, ti, ai, bi)

    # Time axis in seconds
    time = np.arange(len(s)) / sfecg

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 5))

    # ECG waveform
    ax.plot(
        time,
        s,
        color='#B22222',
        linewidth=1.4,
        label='ECG signal'
    )

    # ECG paper-like grid
    ax.set_xticks(np.arange(0, time[-1], 0.2))
    ax.set_xticks(np.arange(0, time[-1], 0.04), minor=True)

    ax.set_yticks(np.arange(-0.4, 1.21, 0.2))
    ax.set_yticks(np.arange(-0.4, 1.21, 0.04), minor=True)

    ax.grid(which='major', color='#E6A0A0', linewidth=0.4)
    ax.grid(which='minor', color='#F5DADA', linewidth=0.2)

    # Identify PQRST peaks
    peak_colors = {
        0: '#2878B5',  # P
        1: '#E69F00',  # Q
        2: '#D62728',  # R
        3: '#2CA02C',  # S
        4: '#9467BD'   # T
    }

    peak_labels = {
        0: 'P',
        1: 'Q',
        2: 'R',
        3: 'S',
        4: 'T'
    }

    for peak_type in range(5):

        indices = np.where(peaks == peak_type)[0]

        ax.scatter(
            time[indices],
            s[indices],
            color=peak_colors[peak_type],
            s=25,
            zorder=5,
            label=peak_labels[peak_type]
        )

    # Labels and title
    ax.set_title(
        'Synthetic Electrocardiogram (ECG)',
        fontsize=17,
        fontweight='bold',
        pad=15
    )

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Normalized ECG amplitude', fontsize=12)

    # Axis limits
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-1.5, 2.5)


    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(
    "synthetic_ECG.png",
    dpi=300,
    bbox_inches="tight"
    )


def interactive_ecg():

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import numpy as np

    # ----------------------------------
    # Initial parameters
    # ----------------------------------

    params = {
        "N": 20,
        "anoise": 0.0,
        "hrmean": 120,
        "hrstd": 1.0,
        "lfhfratio": 0.6,
        "sfecg": 256,
        "sfint": 512
    }

    ai = [1.2, -5, 30, -7.5, 0.75]

    # ----------------------------------
    # Generate initial ECG
    # ----------------------------------

    def generate():

        s, peaks = ecgsyn(
            sfecg=params["sfecg"],
            N=params["N"],
            anoise=params["anoise"],
            hrmean=params["hrmean"],
            hrstd=params["hrstd"],
            lfhfratio=params["lfhfratio"],
            sfint=params["sfint"],
            ai=ai
        )

        t = np.arange(len(s)) / params["sfecg"]

        return t, s, peaks

    t, s, peaks = generate()

    # ----------------------------------
    # Figure
    # ----------------------------------

    fig, ax = plt.subplots(figsize=(15, 7))

    plt.subplots_adjust(
        left=0.12,
        bottom=0.48,
        right=0.95,
        top=0.88
    )

    fig.suptitle(
        "Interactive ECGSYN Simulator",
        fontsize=18,
        fontweight="bold"
    )

    # ----------------------------------
    # ECG plot
    # ----------------------------------

    line, = ax.plot(
        t,
        s,
        color="#B22222",
        linewidth=1.2
    )

    ax.set_title("Synthetic ECG")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (normalized)")

    ax.grid(which="major", color="#E6A0A0")
    ax.grid(which="minor", color="#F5DADA")

    ax.minorticks_on()

    ax.set_ylim(-1, 2)

    # ----------------------------------
    # Peak markers
    # ----------------------------------

    peak_colors = [
        "#2878B5",
        "#E69F00",
        "#D62728",
        "#2CA02C",
        "#9467BD"
    ]

    peak_labels = ["P", "Q", "R", "S", "T"]

    markers = []

    for i in range(5):

        indices = np.where(peaks == i)[0]

        marker = ax.scatter(
            t[indices],
            s[indices],
            color=peak_colors[i],
            s=25,
            label=peak_labels[i]
        )

        markers.append(marker)

    ax.legend(loc="upper right", ncol=5)

    # ----------------------------------
    # Slider creation helper
    # ----------------------------------

    sliders = {}

    def create_slider(name, label, position, minimum, maximum, value):

        slider_ax = fig.add_axes(position)

        slider = Slider(
            slider_ax,
            label,
            minimum,
            maximum,
            valinit=value
        )

        sliders[name] = slider

    # ----------------------------------
    # Physiological parameter sliders
    # ----------------------------------

    create_slider(
        "hrmean", "Heart rate (BPM)",
        [0.15, 0.39, 0.7, 0.025],
        40, 180, 100
    )

    create_slider(
        "hrstd", "HR standard deviation",
        [0.15, 0.34, 0.7, 0.025],
        0, 10, 1
    )

    create_slider(
        "lfhfratio", "LF/HF ratio",
        [0.15, 0.29, 0.7, 0.025],
        0.1, 3, 0.6
    )

    create_slider(
        "anoise", "Noise amplitude",
        [0.15, 0.24, 0.7, 0.025],
        0, 0.5, 0
    )

    # ----------------------------------
    # PQRST amplitude sliders
    # ----------------------------------

    amplitude_ranges = [
        (-5, 5),
        (-15, 0),
        (5, 50),
        (-20, 0),
        (-5, 5)
    ]

    for i, wave in enumerate(peak_labels):

        y = 0.17 - i * 0.035

        create_slider(
            f"wave_{wave}",
            f"{wave} amplitude",
            [0.15, y, 0.7, 0.02],
            amplitude_ranges[i][0],
            amplitude_ranges[i][1],
            ai[i]
        )

    # ----------------------------------
    # Update function
    # ----------------------------------

    def update(value):

        for name in ["hrmean", "hrstd", "lfhfratio", "anoise"]:

            params[name] = sliders[name].val

        for i, wave in enumerate(peak_labels):

            ai[i] = sliders[f"wave_{wave}"].val

        try:

            t, s, peaks = generate()

            line.set_data(t, s)

            # Update peak markers
            for i in range(5):

                indices = np.where(peaks == i)[0]

                markers[i].set_offsets(
                    np.column_stack((t[indices], s[indices]))
                    if len(indices) > 0
                    else np.empty((0, 2))
                )

            ax.set_xlim(0, t[-1])

            ax.set_ylim(
                min(-1, np.min(s) - 0.2),
                max(2, np.max(s) + 0.2)
            )

            fig.canvas.draw_idle()

        except Exception as e:

            print("ECG generation error:", e)

    # ----------------------------------
    # Connect sliders
    # ----------------------------------

    for slider in sliders.values():

        slider.on_changed(update)

    # ----------------------------------
    # Reset button
    # ----------------------------------

    reset_ax = fig.add_axes([0.82, 0.05, 0.12, 0.05])

    reset_button = Button(
        reset_ax,
        "Reset"
    )

    def reset(event):

        sliders["hrmean"].reset()
        sliders["hrstd"].reset()
        sliders["lfhfratio"].reset()
        sliders["anoise"].reset()

        for wave in peak_labels:

            sliders[f"wave_{wave}"].reset()

    reset_button.on_clicked(reset)

    plt.show()




if __name__ == "__main__":
    interactive_ecg()

