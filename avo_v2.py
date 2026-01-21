# ==================================================================================
# AVO-scope: Interactive Two-Layer AVO Visualization Tool
# ==================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ==================================================================================
# ELASTIC PARAMETER FUNCTIONS
# ==================================================================================

def compute_impedance(vp, rho):
    """Compute acoustic impedance with input validation."""
    return np.asarray(vp) * np.asarray(rho)


def compute_vp_vs_ratio(vp, vs):
    """Compute Vp/Vs ratio with zero-division protection."""
    vs = np.asarray(vs)
    return np.asarray(vp) / np.where(vs != 0, vs, 1e-10)


def compute_poisson_ratio(vp, vs):
    """Compute Poisson's ratio from velocities."""
    vpvs = vp / vs
    vpvs_sq = vpvs ** 2
    return 0.5 * (vpvs_sq - 2) / (vpvs_sq - 1)


# ==================================================================================
# RICKER WAVELET
# ==================================================================================

def generate_ricker_wavelet(duration=0.25, dt=0.001, frequency=10):
    """Generate normalized Ricker wavelet (vectorized)."""
    t = np.arange(-duration/2, duration/2, dt)
    pft = (np.pi * frequency * t) ** 2
    wavelet = (1 - 2 * pft) * np.exp(-pft)
    return wavelet / np.abs(wavelet).max()


# ==================================================================================
# ZOEPPRITZ EQUATIONS - Exact Implementation
# ==================================================================================
def compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    Exact Zoeppritz PP reflectivity, adapted from bruges.zoeppritz_rpp.

    Returns real-valued PP reflection coefficients for given angle(s).
    """
    theta1 = np.radians(np.atleast_1d(angles)).astype(complex)

    # Ray parameter
    p = np.sin(theta1) / vp1

    # Angles in lower medium
    theta2 = np.arcsin(p * vp2)   # transmitted P
    phi1   = np.arcsin(p * vs1)   # reflected S
    phi2   = np.arcsin(p * vs2)   # transmitted S

    # Coefficients (Dvorkin et al. formulation) [file:18]
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

    rpp = np.real(rpp)  # for amplitude plots

    return rpp[0] if np.isscalar(angles) else np.asarray(rpp, dtype=float)



def check_critical_angles(vp1, vs1, vp2, vs2, max_angle_deg):
    """
    Check if requested max_angle exceeds P or S critical angles.
    Returns a dict with flags and values in degrees.
    """
    max_angle = np.radians(max_angle_deg)

    # Ray parameter at requested max angle in upper medium
    p = np.sin(max_angle) / vp1

    crit = {}

    # P-wave critical into lower medium (if vp2 < vp1)
    if vp2 < vp1:
        arg_p = vp2 * p
        if np.abs(arg_p) > 1:
            crit['P_critical_exceeded'] = True
        else:
            crit_angle_p = np.degrees(np.arcsin(vp2 / vp1))
            if max_angle_deg > crit_angle_p:
                crit['P_critical_exceeded'] = True
                crit['P_critical_deg'] = crit_angle_p

    # S-wave critical into lower medium (if vs2 < vs1)
    if vs2 < vs1:
        arg_s = vs2 * p
        if np.abs(arg_s) > 1:
            crit['S_critical_exceeded'] = True
        else:
            crit_angle_s = np.degrees(np.arcsin(vs2 / vp1))
            if max_angle_deg > crit_angle_s:
                crit['S_critical_exceeded'] = True
                crit['S_critical_deg'] = crit_angle_s

    return crit

def p_critical_angle_deg(vp1, vp2):
    """
    P-wave critical angle in degrees for incidence from medium 1 to 2.
    Returns None if vp2 >= vp1 (no critical angle).
    """
    if vp2 >= vp1:
        return None
    return np.degrees(np.arcsin(vp2 / vp1))

# ==================================================================================
# SHUEY APPROXIMATIONS (bruges-style)
# ==================================================================================

def compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    2-term Shuey approximation.
    
    R(θ) = R₀ + G·sin²(θ)
    
    Where:
        R₀ = Intercept (normal incidence reflection)
        G  = Gradient (AVO slope)
    """
    theta1 = np.radians(np.atleast_1d(angles).astype(float))
    
    # Compute contrasts and averages
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2) / 2.0
    vs = (vs1 + vs2) / 2.0
    rho = (rho1 + rho2) / 2.0
    
    # Shuey coefficients
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2 / vp**2) * (drho/rho + 2 * dvs/vs)
    
    # Reflection coefficient
    reflection = r0 + g * np.sin(theta1)**2
    
    return r0, g, reflection


def compute_shuey_three_term(vp1, vs1, rho1, vp2, vs2, rho2, angles):
    """
    3-term Shuey approximation with curvature correction.
    
    R(θ) = R₀ + G·sin²(θ) + F·(tan²(θ) - sin²(θ))
    """
    theta1 = np.radians(np.atleast_1d(angles).astype(float))
    
    # Compute contrasts and averages
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2) / 2.0
    vs = (vs1 + vs2) / 2.0
    rho = (rho1 + rho2) / 2.0
    
    # Three-term Shuey coefficients
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2 / vp**2) * (drho/rho + 2 * dvs/vs)
    f = 0.5 * dvp/vp
    
    # Reflection coefficient
    term1 = r0
    term2 = g * np.sin(theta1)**2
    term3 = f * (np.tan(theta1)**2 - np.sin(theta1)**2)
    
    reflection = term1 + term2 + term3
    
    return r0, g, f, reflection


# ==================================================================================
# AVO CLASSIFICATION
# ==================================================================================

def classify_avo_class(intercept, gradient):
    """
    Standard AVO classification (Rutherford & Williams 1989).
    
    Class I:   High impedance (I > 0.05), negative gradient
    Class II:  Near-zero impedance (|I| < 0.05)
    Class IIp: Subset with polarity reversal (negative gradient)
    Class III: Low impedance (I < -0.05), negative gradient
    Class IV:  Low impedance (I < -0.05), positive gradient
    """
    threshold = 0.05
    
    if intercept > threshold:
        return "Class I" if gradient < 0 else "Class I (atypical)"
    elif abs(intercept) <= threshold:
        return "Class IIp" if gradient < 0 else "Class II"
    else:  # intercept < -threshold
        return "Class III" if gradient < 0 else "Class IV"


# ==================================================================================
# SYNTHETIC GATHER GENERATION
# ==================================================================================

def generate_synthetic_gather(vp1, vs1, rho1, vp2, vs2, rho2, 
                             max_angle=30, num_traces=20, 
                             wavelet_freq=10, polarity='normal',
                             method='shuey'):
    """Generate synthetic pre-stack seismic gather."""
    angles = np.linspace(0, max_angle, num_traces)
    
    # Compute AVO response
    if method == 'shuey':
        I, G, avo = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, angles)
    else:
        avo = compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, angles)
    
    # Apply polarity convention
    if polarity == 'reversed':
        avo = -avo
    
    # Generate Ricker wavelet
    wavelet = generate_ricker_wavelet(frequency=wavelet_freq)
    
    # Create synthetic gather
    n_samples = 500
    interface_idx = n_samples // 2
    gather = np.zeros((n_samples, len(angles)))
    
    for i, coeff in enumerate(avo):
        reflection_series = np.zeros(n_samples)
        reflection_series[interface_idx] = coeff
        gather[:, i] = np.convolve(reflection_series, wavelet, mode='same')
    
    return avo, gather, angles


# ==================================================================================
# INTERACTIVE VISUALIZATION
# ==================================================================================

def launch_avoscope():
    """
    Interactive AVO explorer with real-time parameter adjustment.
    """
    
    # Initial parameters (Class III gas sand - Castagna & Backus 1993)
    init_params = {
        'vp1': 3048, 'vs1': 1524, 'rho1': 2.40,  # Shale
        'vp2': 2590, 'vs2': 1060, 'rho2': 2.16,  # Gas sand
        'max_angle': 35
    }
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('AVOscope: Elastic Interface Reflectivity (Zoeppritz & Approximations)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Layout grids
    gs_main = GridSpec(2, 3, figure=fig, left=0.08, right=0.95, 
                      top=0.92, bottom=0.35, hspace=0.3, wspace=0.35)
    gs_sliders = GridSpec(4, 2, figure=fig, left=0.08, right=0.7,
                         top=0.28, bottom=0.05, hspace=0.6, wspace=0.4)
    gs_controls = GridSpec(3, 1, figure=fig, left=0.75, right=0.92,
                          top=0.28, bottom=0.05, hspace=0.3)
    
    # Create axes
    ax_ai = fig.add_subplot(gs_main[0, 0])
    ax_vpvs = fig.add_subplot(gs_main[0, 1])
    ax_gather = fig.add_subplot(gs_main[0, 2])
    ax_avo = fig.add_subplot(gs_main[1, :2])
    ax_crossplot = fig.add_subplot(gs_main[1, 2])
    
    # Create sliders
    slider_params = [
        ('vp1', 'Vp₁ (m/s)', 1500, 5000, init_params['vp1']),
        ('vp2', 'Vp₂ (m/s)', 1500, 5000, init_params['vp2']),
        ('vs1', 'Vs₁ (m/s)', 500, 3000, init_params['vs1']),
        ('vs2', 'Vs₂ (m/s)', 500, 3000, init_params['vs2']),
        ('rho1', 'ρ₁ (g/cm³)', 1.5, 3.0, init_params['rho1']),
        ('rho2', 'ρ₂ (g/cm³)', 1.5, 3.0, init_params['rho2']),
    ]
    
    sliders = {}
    for idx, (param, label, vmin, vmax, vinit) in enumerate(slider_params):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_sliders[row, col])
        sliders[param] = Slider(ax, label, vmin, vmax, valinit=vinit, 
                               valstep=(vmax-vmin)/100)
    
    ax_angle = fig.add_subplot(gs_sliders[3, :])
    sliders['max_angle'] = Slider(ax_angle, 'Max Angle (°)', 10, 50, 
                                  valinit=init_params['max_angle'], valstep=1)
    
    # Control buttons
    ax_method = fig.add_subplot(gs_controls[0, 0])
    radio_method = RadioButtons(ax_method, ('Shuey', 'Zoeppritz'), active=0)
    
    ax_polarity = fig.add_subplot(gs_controls[1, 0])
    radio_polarity = RadioButtons(ax_polarity, ('Normal', 'Reversed'), active=0)
    
    ax_reset = fig.add_subplot(gs_controls[2, 0])
    btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='coral')
    
    def update(val=None):
        # Get parameters
        vp1 = sliders['vp1'].val
        vs1 = sliders['vs1'].val
        rho1 = sliders['rho1'].val
        vp2 = sliders['vp2'].val
        vs2 = sliders['vs2'].val
        rho2 = sliders['rho2'].val
        max_angle = sliders['max_angle'].val
        

        method = radio_method.value_selected.lower()
        polarity = radio_polarity.value_selected.lower()
        
        # --- Critical angle handling ---
        crit = check_critical_angles(vp1, vs1, vp2, vs2, max_angle)

        # Start from slider value
        effective_max_angle = max_angle

        crit_p = p_critical_angle_deg(vp1, vp2)
        if crit_p is not None and max_angle > crit_p:
            effective_max_angle = crit_p
            print("\nWARNING: Requested max angle exceeds P-critical angle.")
            print(f"  P-wave critical angle ≈ {crit_p:.1f}°")
            print("  Angles are limited to this value; beyond it transmitted "
                  "P-waves are evanescent.\n")

        if crit.get('S_critical_exceeded', False) and 'S_critical_deg' in crit:
            print(f"NOTE: S-wave critical angle ≈ {crit['S_critical_deg']:.1f}° "
                  "(mode conversions affected).\n")

        if method == 'shuey' and effective_max_angle > 35:
            print("\nNOTICE: Shuey approximation is typically valid only up to "
                  "≈30–35°. Consider using Zoeppritz for larger angles.\n")
        # max_angle = effective_max_angle

        # Validate Vp > Vs
        if vs1 >= vp1 or vs2 >= vp2:
            return
        
        # Clear axes
        for ax in [ax_ai, ax_vpvs, ax_gather, ax_avo, ax_crossplot]:
            ax.clear()
        
        # PANEL 1: Acoustic Impedance
        n_samples = 500
        z = np.linspace(0, 1, n_samples)
        interface = n_samples // 2
        
        ai = np.concatenate([
            np.full(interface, compute_impedance(vp1, rho1)),
            np.full(n_samples - interface, compute_impedance(vp2, rho2))
        ])
        
        ax_ai.plot(ai, z, 'k-', linewidth=2.5)
        ax_ai.axhline(0.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax_ai.set_xlabel('AI [m/s·g/cm³]', fontweight='bold')
        ax_ai.set_ylabel('Depth', fontweight='bold')
        ax_ai.set_title('Acoustic Impedance', fontweight='bold')
        ax_ai.invert_yaxis()
        ax_ai.grid(alpha=0.3)
        
        # PANEL 2: Vp/Vs Ratio
        vpvs = np.concatenate([
            np.full(interface, compute_vp_vs_ratio(vp1, vs1)),
            np.full(n_samples - interface, compute_vp_vs_ratio(vp2, vs2))
        ])
        
        ax_vpvs.plot(vpvs, z, 'k-', linewidth=2.5)
        ax_vpvs.axhline(0.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax_vpvs.set_xlabel('Vp/Vs', fontweight='bold')
        ax_vpvs.set_ylabel('Depth', fontweight='bold')
        ax_vpvs.set_title('Vp/Vs Ratio', fontweight='bold')
        ax_vpvs.invert_yaxis()
        ax_vpvs.grid(alpha=0.3)
        
        # PANEL 3: Synthetic Gather
        avo_coeffs, gather, angles = generate_synthetic_gather(
            vp1, vs1, rho1, vp2, vs2, rho2,
            max_angle=effective_max_angle, num_traces=15,
            method=method, polarity=polarity
        )
        
        depth_gather = np.linspace(0, 1, gather.shape[0])
        gain = 2.5
        
        for i in range(gather.shape[1]):
            trace = gather[:, i]  / np.max(np.abs(gather)) * gain
            trace_x = i + trace
            
            ax_gather.plot(trace_x, depth_gather, 'k-', linewidth=0.5)
            ax_gather.fill_betweenx(depth_gather, i, trace_x, 
                                   where=(trace_x >= i), 
                                   color='steelblue', alpha=0.5)
            ax_gather.fill_betweenx(depth_gather, i, trace_x, 
                                   where=(trace_x < i), 
                                   color='lightcoral', alpha=0.5)
        
        ax_gather.set_xlabel('Angle (°)', fontweight='bold')
        ax_gather.set_ylabel('Depth', fontweight='bold')
        ax_gather.set_title('Synthetic Gather', fontweight='bold')
        ax_gather.invert_yaxis()
        ax_gather.set_xticks(np.linspace(0, gather.shape[1]-1, 5))
        ax_gather.set_xticklabels([f'{a:.0f}' for a in np.linspace(0, effective_max_angle, 5)])
        
        # PANEL 4: AVO Curves
        angles_fine = np.linspace(0, effective_max_angle, 100)
        
        if method == 'shuey':
            I, G, avo_fine = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, 
                                                   angles_fine)
        else:
            avo_fine = compute_zoeppritz_reflection(vp1, vs1, rho1, vp2, vs2, rho2, 
                                                   angles_fine)
            I, G, _ = compute_shuey_two_term(vp1, vs1, rho1, vp2, vs2, rho2, [0])
        
        if polarity == 'reversed':
            avo_fine = -avo_fine
        
        # Safe scalar extraction
        I_val = I.item() if isinstance(I, np.ndarray) else I
        G_val = G.item() if isinstance(G, np.ndarray) else G
        
        ax_avo.plot(angles_fine, avo_fine, 'k-', linewidth=2.5, label=method.capitalize())
        ax_avo.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax_avo.set_xlabel('Angle (°)', fontweight='bold', fontsize=11)
        ax_avo.set_ylabel('Reflection Coefficient', fontweight='bold', fontsize=11)
        ax_avo.set_title(f'AVO Response ({method.upper()})', fontweight='bold', fontsize=12)
        ax_avo.grid(alpha=0.3)
        ax_avo.legend()
        ax_avo.set_xlim(0, effective_max_angle)
        
         # Grey zone and critical angle annotation AFTER limits are set
        if crit_p is not None and max_angle > effective_max_angle:
            ax_avo.axvspan(effective_max_angle, max_angle,
                           color='lightgray', alpha=0.5,
                           label='> critical angle')

        if crit_p is not None:
            ax_avo.axvline(crit_p, color='red', linestyle='--', alpha=0.7)
            ymax = ax_avo.get_ylim()[1]
            ax_avo.text(crit_p, ymax*0.9,
                        f"θc ≈ {crit_p:.1f}°",
                        rotation=90,
                        color='red',
                        ha='right', va='top',
                        fontsize=9)

        ax_avo.legend()

        # PANEL 5: I-G Crossplot
        xlims, ylims = (-0.4, 0.4), (-0.4, 0.4)
        Ivals = np.linspace(xlims[0], xlims[1], 500)
        fluid_line = -Ivals
        
        # Class regions
        ax_crossplot.add_patch(patches.Rectangle(
            (-0.05, ylims[0]), 0.10, ylims[1]-ylims[0],
            facecolor='cyan', alpha=0.2, edgecolor='none'))
        
        ax_crossplot.fill_between(
            Ivals[Ivals < -0.05], ylims[0], 0,
            color='red', alpha=0.2)
        
        I_class1 = Ivals[Ivals > 0.05]
        G_fluid = -I_class1
        G_min = np.full_like(I_class1, ylims[0])
        ax_crossplot.fill_between(I_class1, G_min, G_fluid, 
                                  color='purple', alpha=0.2)
        
        for I_v in Ivals[Ivals < -0.05]:
            G_top = min(-I_v, ylims[1])
            if G_top > 0:
                ax_crossplot.fill_between([I_v, I_v+0.002], [0, 0], 
                                         [G_top, G_top], 
                                         color='green', alpha=0.15)
        
        ax_crossplot.plot(Ivals, fluid_line, 'k--', lw=1.5, label='Fluid Line')
        ax_crossplot.axhline(0, color='black', lw=1)
        ax_crossplot.axvline(0, color='black', lw=1)
        
        avo_class = classify_avo_class(I_val, G_val)
        ax_crossplot.plot(I_val, G_val, 'o', markersize=14, markerfacecolor='yellow', 
                         markeredgecolor='black', markeredgewidth=2.5, zorder=10)
        
        info_text = f'I: {I_val:.4f}\nG: {G_val:.4f}\n{avo_class}'
        ax_crossplot.text(0.02, 0.98, info_text, transform=ax_crossplot.transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax_crossplot.set_xlim(xlims)
        ax_crossplot.set_ylim(ylims)
        ax_crossplot.set_xlabel('Intercept (I)', fontweight='bold')
        ax_crossplot.set_ylabel('Gradient (G)', fontweight='bold')
        ax_crossplot.set_title('I-G Crossplot', fontweight='bold')
        ax_crossplot.grid(alpha=0.3)
        
        ax_crossplot.text(0.22, -0.35, "I", color="purple", fontsize=11, 
                         style="italic", fontweight="bold")
        ax_crossplot.text(0, -0.37, "II", color="deepskyblue", fontsize=11, 
                         style="italic", fontweight="bold")
        ax_crossplot.text(-0.24, -0.35, "III", color="red", fontsize=11, 
                         style="italic", fontweight="bold")
        ax_crossplot.text(-0.24, 0.22, "IV", color="green", fontsize=11, 
                         style="italic", fontweight="bold")
        
        fig.canvas.draw_idle()
    
    def reset(event):
        for param in ['vp1', 'vs1', 'rho1', 'vp2', 'vs2', 'rho2', 'max_angle']:
            sliders[param].set_val(init_params.get(param, sliders[param].valmin))
        radio_method.set_active(0)
        radio_polarity.set_active(0)
    
    # Connect callbacks
    for slider in sliders.values():
        slider.on_changed(update)
    radio_method.on_clicked(update)
    radio_polarity.on_clicked(update)
    btn_reset.on_clicked(reset)
    
    update()
    plt.show()
    return fig


# ==================================================================================
# USAGE
# ==================================================================================

if __name__ == '__main__':
    launch_avoscope()
