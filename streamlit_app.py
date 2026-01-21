import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CORE PHYSICS FUNCTIONS (from Rock Physics Handbook)
# =========================================================

def compute_impedance(vp, rho):
    return np.asarray(vp) * np.asarray(rho)

def compute_vp_vs_ratio(vp, vs):
    vs = np.asarray(vs)
    return np.asarray(vp) / np.where(vs != 0, vs, 1e-10)

def generate_ricker_wavelet(duration=0.25, dt=0.001, frequency=10):
    t = np.arange(-duration/2, duration/2, dt)
    pft = (np.pi * frequency * t) ** 2
    wavelet = (1 - 2 * pft) * np.exp(-pft)
    return wavelet / np.abs(wavelet).max()

def compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    theta1 = np.radians(np.atleast_1d(angles)).astype(complex)
    p = np.sin(theta1) / vp1
    theta2 = np.arcsin(p * vp2)   # P transmitted
    phi1   = np.arcsin(p * vs1)   # S reflected
    phi2   = np.arcsin(p * vs2)   # S transmitted

    a = rho2 * (1 - 2 * np.sin(phi2)**2) - rho1 * (1 - 2 * np.sin(phi1)**2)
    b = rho2 * (1 - 2 * np.sin(phi2)**2) + 2 * rho1 * np.sin(phi1)**2
    c = rho1 * (1 - 2 * np.sin(phi1)**2) + 2 * rho2 * np.sin(phi2)**2
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p**2

    rpp = (1 / D) * (
        F * (b * (np.cos(theta1) / vp1) - c * (np.cos(theta2) / vp2))
        - H * p**2 * (a + d * (np.cos(theta1) / vp1) * (np.cos(phi2) / vs2))
    )

    rpp = np.real(rpp)
    return rpp[0] if np.isscalar(angles) else np.asarray(rpp, dtype=float)

def check_critical_angles(vp1, vs1, vp2, vs2, max_angle_deg):
    max_angle = np.radians(max_angle_deg)
    p = np.sin(max_angle) / vp1
    crit = {}

    if vp2 < vp1:
        arg_p = vp2 * p
        if np.abs(arg_p) > 1:
            crit["P_critical_exceeded"] = True
        else:
            crit_angle_p = np.degrees(np.arcsin(vp2 / vp1))
            if max_angle_deg > crit_angle_p:
                crit["P_critical_exceeded"] = True
                crit["P_critical_deg"] = crit_angle_p

    if vs2 < vs1:
        arg_s = vs2 * p
        if np.abs(arg_s) > 1:
            crit["S_critical_exceeded"] = True
        else:
            crit_angle_s = np.degrees(np.arcsin(vs2 / vp1))
            if max_angle_deg > crit_angle_s:
                crit["S_critical_exceeded"] = True
                crit["S_critical_deg"] = crit_angle_s

    return crit

def p_critical_angle_deg(vp1, vp2):
    if vp2 >= vp1:
        return None
    return np.degrees(np.arcsin(vp2 / vp1))

def compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    theta1 = np.radians(np.atleast_1d(angles).astype(float))
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2) / 2.0
    vs = (vs1 + vs2) / 2.0
    rho = (rho1 + rho2) / 2.0
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2 / vp**2) * (drho/rho + 2 * dvs/vs)
    reflection = r0 + g * np.sin(theta1)**2
    return r0, g, reflection

def compute_shuey_three_term(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    theta1 = np.radians(np.atleast_1d(angles).astype(float))
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2) / 2.0
    vs = (vs1 + vs2) / 2.0
    rho = (rho1 + rho2) / 2.0
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2 / vp**2) * (drho/rho + 2 * dvs/vs)
    f = 0.5 * dvp/vp
    term1 = r0
    term2 = g * np.sin(theta1)**2
    term3 = f * (np.tan(theta1)**2 - np.sin(theta1)**2)
    reflection = term1 + term2 + term3
    return r0, g, f, reflection

def classify_avo_class(intercept, gradient):
    threshold = 0.05
    if intercept > threshold:
        return "Class I" if gradient < 0 else "Class I (atypical)"
    elif abs(intercept) <= threshold:
        return "Class IIp" if gradient < 0 else "Class II"
    else:
        return "Class III" if gradient < 0 else "Class IV"

def generate_synthetic_gather(vp1, vs1, rho1, vp2, vs2, rho2,
                              max_angle=30, num_traces=20,
                              wavelet_freq=10, polarity="normal",
                              method="shuey"):
    angles = np.linspace(0, max_angle, num_traces)
    if method.startswith("shuey"):
        I, G, avo = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, angles)
    else:
        avo = compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, angles)

    if polarity == "reversed":
        avo = -avo

    wavelet = generate_ricker_wavelet(frequency=wavelet_freq)
    n_samples = 500
    interface_idx = n_samples // 2
    gather = np.zeros((n_samples, len(angles)))
    for i, coeff in enumerate(avo):
        reflection_series = np.zeros(n_samples)
        reflection_series[interface_idx] = coeff
        gather[:, i] = np.convolve(reflection_series, wavelet, mode="same")
    return avo, gather, angles

# =========================================================
# RELATIVE ERROR PRESETS (AI contrast cases)
# =========================================================

# Each case is (vp1, vs1, rho1, vp2, vs2, rho2)
ERROR_PRESETS = {
    "Weak contrast (|R0| ≲ 0.05)": dict(
        vp1=3000, vs1=1500, rho1=2.35,
        vp2=3100, vs2=1550, rho2=2.40
    ),
    "Moderate contrast (|R0| ≈ 0.1)": dict(
        vp1=3000, vs1=1500, rho1=2.35,
        vp2=2700, vs2=1350, rho2=2.20
    ),
    "Strong contrast (|R0| ≳ 0.2)": dict(
        vp1=3000, vs1=1500, rho1=2.35,
        vp2=2300, vs2=1100, rho2=2.00
    ),
}


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(page_title="AVOscope", layout="wide")

# ---- Header ----
st.title("AVOscope — Elastic AVO & Reflectivity Explorer")
st.caption("Exact Zoeppritz vs AVO approximations for a two-layer isotropic interface")

with st.expander("Model assumptions & disclaimer"):
    st.markdown(
        "- Plane-wave Zoeppritz reflectivity\n"
        "- Two homogeneous, isotropic elastic half-spaces\n"
        "- No attenuation, no multiples, no tuning\n"
        "- Amplitudes are displacement reflection coefficients\n"
        "- Post-critical angles involve evanescent waves and are not strictly interpretable"
    )

# ---- Presets ----
st.sidebar.subheader("AVO Class Preset")
preset = st.sidebar.selectbox(
    "Select scenario",
    ["Custom", "Class I", "Class II", "Class IIp", "Class III", "Class IV"],
)

# Simple, illustrative preset values (tunable later)
preset_params = {
    "Class I":  dict(vp1=2000, vs1=1155, rho1=2.40, vp2=4000, vs2=2300, rho2=2.66),
    "Class IIp": dict(vp1=1800, vs1=666, rho1=2.40, vp2=2400, vs2=1700, rho2=1.95),
    "Class II":dict(vp1=3000, vs1=1500, rho1=2.35, vp2=2950, vs2=1480, rho2=2.33),
    "Class III":dict(vp1=2200, vs1=820, rho1=2.16, vp2=1550, vs2=900, rho2=1.80),
    "Class IV": dict(vp1=3240, vs1=1620, rho1=2.34, vp2=1650, vs2=820,  rho2=2.00),
}

if preset != "Custom":
    params = preset_params[preset]
else:
    params = preset_params["Class III"]  # default starting point

lock_layers = (preset != "Custom")
if lock_layers:
    st.sidebar.info("Preset active: layer sliders locked. Choose 'Custom' to edit.")

# ---- Sidebar: layer properties ----
st.sidebar.subheader("Upper layer (1)")
col1, col2 = st.sidebar.columns(2)
with col1:
    vp1 = st.number_input("Vp₁ (m/s)", 1500, 6000, params["vp1"], 50, disabled=lock_layers)
    vs1 = st.number_input("Vs₁ (m/s)", 500, 4000, params["vs1"], 50, disabled=lock_layers)
with col2:
    rho1 = st.number_input("ρ₁ (g/cm³)", 1.5, 3.0, float(params["rho1"]), 0.05, disabled=lock_layers)

st.sidebar.subheader("Lower layer (2)")
col3, col4 = st.sidebar.columns(2)
with col3:
    vp2 = st.number_input("Vp₂ (m/s)", 1500, 6000, params["vp2"], 50, disabled=lock_layers)
    vs2 = st.number_input("Vs₂ (m/s)", 500, 4000, params["vs2"], 50, disabled=lock_layers)
with col4:
    rho2 = st.number_input("ρ₂ (g/cm³)", 1.5, 3.0, float(params["rho2"]), 0.05, disabled=lock_layers)

# ---- Sidebar: angle / method / polarity ----
st.sidebar.subheader("Angle & method")
max_angle = st.sidebar.slider("Max incidence angle (°)", 5, 50, 35, 1)
method_choice = st.sidebar.radio(
    "Reflectivity method",
    ["Shuey (2-term)", "Shuey (3-term)", "Zoeppritz"],
)
polarity = st.sidebar.radio("Polarity convention", ["Normal", "Reversed"])

auto_limit = True  # could be a checkbox later if you want

# ---- Basic sanity checks ----
if vs1 >= vp1 or vs2 >= vp2:
    st.error("Vs must be less than Vp in both layers (physical constraint).")
    st.stop()

# ---- Physics warnings & effective max angle ----
crit = check_critical_angles(vp1, vs1, vp2, vs2, max_angle)
crit_p = p_critical_angle_deg(vp1, vp2)
effective_max_angle = max_angle

if crit_p is not None and max_angle > crit_p and auto_limit:
    effective_max_angle = crit_p
    st.warning(
        f"Max angle exceeds P-wave critical angle (θc ≈ {crit_p:.1f}°). "
        "Angles are limited to this value; beyond it transmitted P-waves are evanescent."
    )

if method_choice.startswith("Shuey") and effective_max_angle > 35:
    st.warning("Shuey approximation is usually reliable only up to about 30–35°. "
               "Consider 'Zoeppritz' for larger angles.")

if crit.get("S_critical_exceeded", False) and "S_critical_deg" in crit:
    st.info(f"S-wave critical angle ≈ {crit['S_critical_deg']:.1f}°. "
            "Mode conversions are strongly affected near and beyond this angle.")

# =========================================================
# TABS
# =========================================================

tab_logs, tab_avo, tab_gather, tab_ig, tab_err = st.tabs(
    ["Elastic logs", "AVO curve", "Synthetic gather", "I–G crossplot", "Error analysis"]
)

# ---- Common quantities ----
Z1 = compute_impedance(vp1, rho1)
Z2 = compute_impedance(vp2, rho2)
ai_contrast = (Z2 - Z1) / (Z2 + Z1)

# For I,G always use Shuey 2-term at θ=0
I0, G0, _ = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, [0])
I_scalar = float(np.squeeze(I0))
G_scalar = float(np.squeeze(G0))
avo_class = classify_avo_class(I_scalar, G_scalar)

# ---- Tab 1: Elastic logs ----
with tab_logs:
    st.subheader("Elastic property logs")

    n_samples = 500
    z = np.linspace(0, 1, n_samples)
    interface = n_samples // 2

    ai_log = np.concatenate([
        np.full(interface, Z1),
        np.full(n_samples - interface, Z2),
    ])
    vpvs_log = np.concatenate([
        np.full(interface, compute_vp_vs_ratio(vp1, vs1)),
        np.full(n_samples - interface, compute_vp_vs_ratio(vp2, vs2)),
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    ax1.plot(ai_log, z, "k-", lw=2.5)
    ax1.axhline(0.5, color="red", ls="--", alpha=0.6, lw=1.5)
    ax1.set_xlabel("AI [m/s·g/cm³]")
    ax1.set_ylabel("Depth (arb.)")
    ax1.set_title("Acoustic impedance")
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3)

    ax2.plot(vpvs_log, z, "k-", lw=2.5)
    ax2.axhline(0.5, color="red", ls="--", alpha=0.6, lw=1.5)
    ax2.set_xlabel("Vp/Vs")
    ax2.set_ylabel("Depth (arb.)")
    ax2.set_title("Vp/Vs ratio")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        f"- AI contrast (normal incidence R): **{ai_contrast:.3f}**  \n"
        f"- Shuey intercept I₀: **{I_scalar:.3f}**, gradient G: **{G_scalar:.3f}**  \n"
        f"- AVO class (Rutherford & Williams): **{avo_class}**"
    )

# ---- Tab 2: AVO curve ----
with tab_avo:
    st.subheader("Angle-dependent reflectivity")

    angles_fine = np.linspace(0, effective_max_angle, 200)
    

    if method_choice == "Shuey (2-term)":
        I, G, R = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, angles_fine)
    elif method_choice == "Shuey (3-term)":
        I, G, F, R = compute_shuey_three_term(vp1, vs1, rho1, vp2, vs2, rho2, angles_fine)
    else:
        R = compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, angles_fine)

    if polarity == "Reversed":
        R = -R


    fig2, ax = plt.subplots(figsize=(7, 5))
    style = "-" if method_choice == "Zoeppritz" else "--"
    ax.plot(angles_fine, R, style, color="k", lw=2.5, label=method_choice)
    ax.axhline(0, color="gray", lw=1)

    if crit_p is not None:
        ax.axvline(crit_p, color="red", ls="--", alpha=0.7)
        ymax = ax.get_ylim()[1]
        ax.text(crit_p, ymax * 0.9,
                f"θc ≈ {crit_p:.1f}°",
                rotation=90, color="red",
                ha="right", va="top", fontsize=9)

    if crit_p is not None and max_angle > effective_max_angle:
        ax.axvspan(effective_max_angle, max_angle,
                   color="lightgray", alpha=0.5,
                   label="> critical angle")

    ax.set_xlim(0, max_angle)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Reflection coefficient")
    ax.set_title("AVO response")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig2)

# ---- Tab 3: Synthetic gather ----
with tab_gather:
    st.subheader("Illustrative pre-stack synthetic gather")
    avo_coeffs, gather, angles = generate_synthetic_gather(
        vp1, vs1, rho1, vp2, vs2, rho2,
        max_angle=effective_max_angle, num_traces=15,
        method=method_choice.lower().split()[0], polarity=polarity.lower()
    )

    fig3, axg = plt.subplots(figsize=(7, 6))
    depth = np.linspace(0, 1, gather.shape[0])
    gain = 1.5

    for i in range(gather.shape[1]):
        trace = gather[:, i] / np.max(np.abs(avo_coeffs)) * gain
        x = i + trace
        axg.plot(x, depth, "k-", lw=0.5)
        axg.fill_betweenx(depth, i, x, where=(x >= i),
                          color="steelblue", alpha=0.5)
        axg.fill_betweenx(depth, i, x, where=(x < i),
                          color="lightcoral", alpha=0.5)

    axg.set_ylim(depth.max(), depth.min())
    axg.set_xlabel("Angle (°)")
    axg.set_ylabel("Depth (arb.)")
    axg.set_title("Synthetic gather (illustrative, not amplitude-preserved)")
    axg.set_xticks(np.linspace(0, gather.shape[1]-1, 5))
    axg.set_xticklabels([f"{a:.0f}" for a in np.linspace(0, effective_max_angle, 5)])
    axg.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)

# ---- Tab 4: I–G crossplot ----
with tab_ig:
    st.subheader("Intercept–gradient crossplot")

    xlims, ylims = (-0.5, 0.5), (-0.5, 0.5)
    Ivals = np.linspace(xlims[0], xlims[1], 500)
    fluid_line = -Ivals

    fig4, axig = plt.subplots(figsize=(7, 7))

    axig.add_patch(patches.Rectangle(
        (-0.05, ylims[0]), 0.10, ylims[1]-ylims[0],
        facecolor="cyan", alpha=0.2, edgecolor="none"
    ))

    axig.fill_between(Ivals[Ivals < -0.05], ylims[0], 0,
                      color="red", alpha=0.2)

    I_class1 = Ivals[Ivals > 0.05]
    G_fluid = -I_class1
    G_min = np.full_like(I_class1, ylims[0])
    axig.fill_between(I_class1, G_min, G_fluid, color="purple", alpha=0.2)

    for I_v in Ivals[Ivals < -0.05]:
        G_top = min(-I_v, ylims[1])
        if G_top > 0:
            axig.fill_between([I_v, I_v+0.002], [0, 0], [G_top, G_top],
                              color="green", alpha=0.15)

    axig.plot(Ivals, fluid_line, "k--", lw=1.5, label="Fluid line")
    axig.axhline(0, color="black", lw=1)
    axig.axvline(0, color="black", lw=1)

    axig.plot(I_scalar, G_scalar, "o", markersize=14,
              markerfacecolor="yellow", markeredgecolor="black",
              markeredgewidth=2.5, zorder=10)

    info_text = f"I: {I_scalar:.4f}\nG: {G_scalar:.4f}\nClass: {avo_class}"
    axig.text(0.02, 0.98, info_text, transform=axig.transAxes,
              fontsize=9, va="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    axig.set_xlim(xlims)
    axig.set_ylim(ylims)
    axig.set_xlabel("Intercept (I)")
    axig.set_ylabel("Gradient (G)")
    axig.set_title("Rutherford & Williams AVO classes")
    axig.grid(alpha=0.3)

    axig.text(0.22, -0.35, "I", color="purple", fontsize=11,
              style="italic", fontweight="bold")
    axig.text(0, -0.37, "II", color="deepskyblue", fontsize=11,
              style="italic", fontweight="bold")
    axig.text(-0.24, -0.35, "III", color="red", fontsize=11,
              style="italic", fontweight="bold")
    axig.text(-0.24, 0.22, "IV", color="green", fontsize=11,
              style="italic", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig4)

# =========================================================
# RELATIVE ERROR VS ANGLE (Zoeppritz vs Shuey)
# =========================================================

with tab_err:
    st.subheader("Relative error: Zoeppritz vs Shuey (by contrast case)")
    st.markdown(
        "Select a canonical interface to see how well "
        "Shuey 2-term and 3-term approximate the exact Zoeppritz PP reflectivity."
    )

    case = st.selectbox(
        "Acoustic impedance contrast case",
        list(ERROR_PRESETS.keys()),
        index=1  # default: Moderate
    )

    preset_err = ERROR_PRESETS[case]
    vp1_e, vs1_e, rho1_e = preset_err["vp1"], preset_err["vs1"], preset_err["rho1"]
    vp2_e, vs2_e, rho2_e = preset_err["vp2"], preset_err["vs2"], preset_err["rho2"]

    # Compute R0 of this case for info
    Z1_e = compute_impedance(vp1_e, rho1_e)
    Z2_e = compute_impedance(vp2_e, rho2_e)
    R0_ai = (Z2_e - Z1_e) / (Z2_e + Z1_e)

    st.markdown(
        f"- Upper: Vp₁ = {vp1_e} m/s, Vs₁ = {vs1_e} m/s, ρ₁ = {rho1_e:.2f} g/cm³  \n"
        f"- Lower: Vp₂ = {vp2_e} m/s, Vs₂ = {vs2_e} m/s, ρ₂ = {rho2_e:.2f} g/cm³  \n"
        f"- Normal-incidence reflection (AI): **R₀ ≈ {R0_ai:.3f}**"
    )

    # Angle range for error analysis (stay within reasonable angles)
    max_angle_err = min(effective_max_angle, 45)  # avoid too large if user sets 50°
    angles_err = np.linspace(0, max_angle_err, 200)

    # Exact Zoeppritz
    Rz = compute_zoeppritz_reflection(
        vp1_e, vs1_e, rho1_e, vp2_e, vs2_e, rho2_e, angles_err
    )

    # Shuey 2-term and 3-term
    I2, G2, Rs2 = compute_shuey_two_term(
        vp1_e, vs1_e, rho1_e, vp2_e, vs2_e, rho2_e, angles_err
    )
    I3, G3, F3, Rs3 = compute_shuey_three_term(
        vp1_e, vs1_e, rho1_e, vp2_e, vs2_e, rho2_e, angles_err
    )

    # Use the same polarity convention as main app for consistency
    if polarity == "Reversed":
        Rz, Rs2, Rs3 = -Rz, -Rs2, -Rs3

    # Absolute error vs angle
    err2 = Rz - Rs2
    err3 = Rz - Rs3

    fig_err, axe = plt.subplots(figsize=(7, 4))
    axe.axhline(0, color="gray", lw=1)

    axe.plot(angles_err, err2, "--", color="tab:blue",
             lw=2, label="Zoeppritz − Shuey 2-term")
    axe.plot(angles_err, err3, "-.", color="tab:orange",
             lw=2, label="Zoeppritz − Shuey 3-term")

    # Optional: relative error in % at larger angles
    rel2 = np.where(Rz != 0, err2 / Rz * 100, 0)
    rel3 = np.where(Rz != 0, err3 / Rz * 100, 0)

    crit_p_case = p_critical_angle_deg(vp1_e, vp2_e)
    if crit_p_case is not None:
        axe.axvline(crit_p_case, color="red", ls="--", alpha=0.7)
        ymax = axe.get_ylim()[1]
        axe.text(crit_p_case, ymax * 0.9,
                 f"θc ≈ {crit_p_case:.1f}°",
                 rotation=90, color="red",
                 ha="right", va="top", fontsize=9)

    axe.set_xlabel("Angle (°)")
    axe.set_ylabel("ΔR (Zoeppritz − Shuey)")
    axe.set_title("Approximation error vs angle")
    axe.grid(alpha=0.3)
    axe.legend()
    plt.tight_layout()
    st.pyplot(fig_err)

    st.markdown(
        "- Near 0°, all curves agree (errors ≈ 0).\n"
        "- As contrast strengthens and angle increases, the 3-term Shuey "
        "usually tracks Zoeppritz better than the 2-term."
    )


