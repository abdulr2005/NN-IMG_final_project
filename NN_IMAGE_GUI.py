import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
from scipy.ndimage import gaussian_filter

# ══════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════
C = {
    "bg":       "#090b10",
    "sidebar":  "#0d1018",
    "card":     "#111520",
    "card2":    "#181d2c",
    "border":   "#252d42",
    "div":      "#00e5ff",      # cyan separator lines
    "accent":   "#00e5ff",
    "red":      "#ff3d71",
    "green":    "#39d353",
    "warn":     "#ffb300",
    "purple":   "#b57bee",
    "text":     "#dce3f5",
    "text2":    "#4e587a",
    "text3":    "#8a96bc",
    "btn":      "#00e5ff",
    "btn_fg":   "#090b10",
    "btn_h":    "#00bcd4",
    "tab_act":  "#00e5ff",
    "tab_in":   "#1a1e2c",
}
# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════
FORGERY_THRESHOLD = 2.5  # تقدر تغير النسبة دي، وهتتسمع في الشاشة كلها (النص والرسمة)

MF  = ("Courier New", 9)
MFB = ("Courier New", 9,  "bold")
MFL = ("Courier New", 8)
MFT = ("Courier New", 11, "bold")

# ══════════════════════════════════════════════════════════════════════
#  KERAS OBJECTS
# ══════════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0):
    a = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    b = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    i = tf.reduce_sum(a * b)
    return (2.*i + smooth) / (tf.reduce_sum(a) + tf.reduce_sum(b) + smooth)

@tf.keras.utils.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1 - dice_coef(y_true, y_pred)) * 2

@tf.keras.utils.register_keras_serializable()
def total_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1 - dice_coef(y_true, y_pred))

# ══════════════════════════════════════════════════════════════════════
#  FORENSIC EXTRACTORS
# ══════════════════════════════════════════════════════════════════════
def get_ela(path, quality=90):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    tmp = "_ela_tmp.jpg"
    cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    rs  = cv2.imread(tmp)
    ela = cv2.absdiff(img, rs)
    ela = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    if os.path.exists(tmp): os.remove(tmp)
    return np.expand_dims(ela.astype(np.float32) / 255.0, -1)

def get_noise(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    den = gaussian_filter(img, sigma=2)
    nm  = img - den
    nm  = (nm - nm.min()) / (nm.max() - nm.min() + 1e-7)
    return np.expand_dims(nm, -1)

def get_edges(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    ed  = cv2.Canny(img, 100, 200)
    return np.expand_dims(ed.astype(np.float32) / 255.0, -1)

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def hex_to_rgb01(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def pil_thumb(path, w, h, radius=10):
    img  = Image.open(path).resize((w, h), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w-1, h-1], radius=radius, fill=255)
    bg   = Image.new("RGBA", (w, h), (17, 21, 32, 255))
    bg.paste(img, mask=mask)
    return ImageTk.PhotoImage(bg)

def sep(parent, color=None, thick=1, orient="h", pad=0):
    color = color or C["border"]
    if orient == "h":
        tk.Frame(parent, bg=color, height=thick).pack(fill=tk.X, pady=pad)
    else:
        tk.Frame(parent, bg=color, width=thick).pack(fill=tk.Y, padx=pad)

def info_row(parent, key, val, bg):
    f = tk.Frame(parent, bg=bg)
    f.pack(fill=tk.X, padx=10, pady=2)
    tk.Label(f, text=f"{key}:", fg=C["text2"], bg=bg, font=MF,
             width=12, anchor="w").pack(side=tk.LEFT)
    lbl = tk.Label(f, text=val, fg=C["text"], bg=bg, font=MFB, anchor="w")
    lbl.pack(side=tk.LEFT)
    return lbl

def section_hdr(parent, title, bg):
    tk.Frame(parent, bg=bg, height=10).pack()
    row = tk.Frame(parent, bg=bg)
    row.pack(fill=tk.X, padx=14)
    tk.Frame(row, bg=C["div"], width=3, height=14).pack(side=tk.LEFT)
    tk.Label(row, text=f"  {title}", fg=C["accent"], bg=bg,
             font=("Courier New", 8, "bold")).pack(side=tk.LEFT)
    tk.Frame(parent, bg=C["border"], height=1).pack(fill=tk.X, padx=14, pady=(4, 6))

# ══════════════════════════════════════════════════════════════════════
#  CUSTOM TAB BAR  (replaces ttk.Notebook for full style control)
# ══════════════════════════════════════════════════════════════════════
class TabBar(tk.Frame):
    def __init__(self, parent, tabs, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self._frames  = {}
        self._buttons = {}
        self._active  = None
        self._content = None          # set after build

        for name in tabs:
            btn = tk.Button(
                self, text=name,
                bg=C["card"], fg=C["text2"],
                font=MFB, relief="flat", bd=0,
                cursor="hand2", padx=20, pady=8,
                activebackground=C["card2"],
                activeforeground=C["accent"],
                command=lambda n=name: self.show(n)
            )
            btn.pack(side=tk.LEFT)
            self._buttons[name] = btn

        # right filler
        tk.Frame(self, bg=C["card"]).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def attach_content(self, frame):
        self._content = frame
        for name in self._buttons:
            f = tk.Frame(self._content, bg=C["bg"])
            self._frames[name] = f
        # show first
        first = next(iter(self._buttons))
        self.show(first)

    def show(self, name):
        for n, f in self._frames.items():
            f.pack_forget()
        self._frames[name].pack(fill=tk.BOTH, expand=True)

        for n, b in self._buttons.items():
            if n == name:
                b.config(fg=C["accent"], bg=C["card2"])
            else:
                b.config(fg=C["text2"], bg=C["card"])
        self._active = name

    def frame(self, name):
        return self._frames[name]


# ══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════
class ForgeryDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MUST AI Forensics  ·  Forgery Detector")
        self.root.geometry("1620x960")
        self.root.minsize(1200, 780)
        self.root.configure(bg=C["bg"])

        self.model_path = r'C:\Users\abdul\Desktop\NN+IMG_PROJECT\ultimate_forgery_model_v2.keras'
        self.model = None
        self._refs = []
        self._fig1 = None   # images tab figure
        self._fig2 = None   # charts tab figure

        self._load_model()
        self._build()

    # ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'hybrid_loss': hybrid_loss,
                    'total_loss':  total_loss,
                    'dice_coef':   dice_coef,
                },
                compile=False
            )
            print("✅ Model loaded.")
        except Exception as e:
            print(f"⚠️  {e}")

    # ══════════════════════════════════════════════════════════════════
    #  LAYOUT
    # ══════════════════════════════════════════════════════════════════
    def _build(self):
        self._build_topbar()

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)

        # cyan vertical divider between sidebar and main
        self._build_sidebar(body)
        tk.Frame(body, bg=C["div"], width=2).pack(side=tk.LEFT, fill=tk.Y)
        self._build_main(body)

    # ── TOP BAR ───────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=C["sidebar"], height=54)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Frame(bar, bg=C["div"], width=4).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(bar, text=" ⬡ MUST AI ", fg=C["accent"], bg=C["sidebar"],
                 font=("Courier New", 15, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(bar, text="FORENSICS LAB", fg=C["text2"], bg=C["sidebar"],
                 font=MF).pack(side=tk.LEFT, pady=18)

        bbg = C["green"] if self.model else C["red"]
        btx = "  ● MODEL READY  " if self.model else "  ● MODEL NOT FOUND  "
        tk.Label(bar, text=btx, fg=C["btn_fg"], bg=bbg,
                 font=MFB, padx=8, pady=4).pack(side=tk.RIGHT, padx=20, pady=14)

        tk.Frame(self.root, bg=C["div"], height=1).pack(fill=tk.X)

    # ── SIDEBAR ───────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["sidebar"], width=272)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        tk.Frame(sb, bg=C["sidebar"], height=14).pack()

        # upload button
        self._btn = tk.Button(
            sb, text="  ↑  UPLOAD IMAGE  ",
            command=self._start_analysis,
            bg=C["btn"], fg=C["btn_fg"],
            font=("Courier New", 10, "bold"),
            relief="flat", bd=0, cursor="hand2",
            padx=8, pady=10,
            activebackground=C["btn_h"],
            activeforeground=C["btn_fg"]
        )
        self._btn.pack(padx=16, fill=tk.X)
        tk.Frame(sb, bg=C["sidebar"], height=10).pack()

        # thumbnail
        section_hdr(sb, "ORIGINAL IMAGE", C["sidebar"])
        prev = tk.Frame(sb, bg=C["card"],
                        highlightbackground=C["border"], highlightthickness=1)
        prev.pack(padx=14, pady=(0, 4), fill=tk.X)
        self._thumb = tk.Label(prev, bg=C["card"], text="—",
                               fg=C["text2"], font=MF, width=30, height=9)
        self._thumb.pack()

        # file info
        section_hdr(sb, "FILE INFO", C["sidebar"])
        ic = tk.Frame(sb, bg=C["card"],
                      highlightbackground=C["border"], highlightthickness=1)
        ic.pack(padx=14, pady=(0, 4), fill=tk.X)
        tk.Frame(ic, bg=C["card"], height=4).pack()
        self._iname = info_row(ic, "Name",       "—", C["card"])
        self._idims = info_row(ic, "Dimensions", "—", C["card"])
        self._isize = info_row(ic, "File size",  "—", C["card"])
        tk.Frame(ic, bg=C["card"], height=6).pack()

        # verdict
        section_hdr(sb, "VERDICT", C["sidebar"])
        self._vc = tk.Frame(sb, bg=C["card2"],
                            highlightbackground=C["border"],
                            highlightthickness=1, height=88)
        self._vc.pack(padx=14, pady=(0, 4), fill=tk.X)
        self._vc.pack_propagate(False)
        self._vl = tk.Label(self._vc, text="AWAITING\nANALYSIS",
                            fg=C["text2"], bg=C["card2"],
                            font=("Courier New", 10, "bold"))
        self._vl.place(relx=0.5, rely=0.5, anchor="center")

        # progress
        section_hdr(sb, "PROGRESS", C["sidebar"])
        pf = tk.Frame(sb, bg=C["sidebar"])
        pf.pack(padx=14, fill=tk.X)

        sty = ttk.Style()
        sty.theme_use('clam')
        sty.configure("X.Horizontal.TProgressbar",
                      troughcolor=C["card"], background=C["accent"],
                      bordercolor=C["border"],
                      lightcolor=C["accent"], darkcolor=C["accent"])
        self._prog = ttk.Progressbar(pf, style="X.Horizontal.TProgressbar",
                                     orient="horizontal", mode="determinate")
        self._prog.pack(fill=tk.X)
        self._slbl = tk.Label(pf, text="Ready", fg=C["text2"],
                              bg=C["sidebar"], font=MFL)
        self._slbl.pack(anchor="w", pady=(3, 0))

    # ── MAIN (tabs) ───────────────────────────────────────────────────
    def _build_main(self, parent):
        main = tk.Frame(parent, bg=C["bg"])
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # tab bar
        self._tabs = TabBar(main, ["  🖼  FORENSIC IMAGES  ", "  📊  ANALYSIS CHARTS  "])
        self._tabs.pack(fill=tk.X)

        # thin cyan underline below tab bar
        tk.Frame(main, bg=C["div"], height=2).pack(fill=tk.X)

        # content area
        self._content = tk.Frame(main, bg=C["bg"])
        self._content.pack(fill=tk.BOTH, expand=True)

        self._tabs.attach_content(self._content)

        # placeholders
        for tab_name, msg in [
            ("  🖼  FORENSIC IMAGES  ", "Upload an image to view forensic channels"),
            ("  📊  ANALYSIS CHARTS  ", "Analysis charts will appear here after processing"),
        ]:
            tk.Label(
                self._tabs.frame(tab_name),
                text=msg, fg=C["text2"], bg=C["bg"],
                font=("Courier New", 13)
            ).place(relx=0.5, rely=0.5, anchor="center")

        self._tab_img = "  🖼  FORENSIC IMAGES  "
        self._tab_ch  = "  📊  ANALYSIS CHARTS  "

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE
    # ══════════════════════════════════════════════════════════════════
    def _set_status(self, msg, val=0):
        self._slbl.config(text=msg)
        self._prog['value'] = val
        self.root.update_idletasks()

    def _start_analysis(self):
        fp = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not fp: return
        threading.Thread(target=self._run, args=(fp,), daemon=True).start()

    def _run(self, fp):
        try:
            self._btn.config(state="disabled")

            orig = cv2.imread(fp)
            h, w = orig.shape[:2]
            fname = os.path.basename(fp)
            fsize = os.path.getsize(fp)
            fs    = f"{fsize/1024:.1f} KB" if fsize < 1e6 else f"{fsize/1e6:.2f} MB"

            self._iname.config(text=(fname[:22]+"…" if len(fname)>22 else fname))
            self._idims.config(text=f"{w} × {h} px")
            self._isize.config(text=fs)

            ph = pil_thumb(fp, 240, 152)
            self._refs.append(ph)
            self._thumb.config(image=ph, text="", height=152)

            self._set_status("Extracting ELA…", 18)
            ela   = get_ela(fp)
            self._set_status("Extracting noise residual…", 36)
            noise = get_noise(fp)
            self._set_status("Computing edge map…", 54)
            edge  = get_edges(fp)

            rgb = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
                             (256, 256)) / 255.0
            inp = np.concatenate([rgb, ela, noise, edge], axis=-1)

            if self.model:
                self._set_status("Running neural network…", 72)
                pred = self.model.predict(np.expand_dims(inp, 0), verbose=0)[0]
            else:
                pred = np.zeros((256, 256, 1), dtype=np.float32)

            self._set_status("Rendering…", 88)
            orig_pil = Image.open(fp)
            self.root.after(0, lambda: self._render(fp, orig_pil, rgb, ela, noise, edge, pred))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self._btn.config(state="normal")
            self._set_status("Error", 0)

    # ══════════════════════════════════════════════════════════════════
    #  RENDER
    # ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
    #  RENDER
    # ══════════════════════════════════════════════════════════════════════
    def _render(self, fp, orig_pil, rgb, ela, noise, edge, pred):
        mask       = (pred.squeeze() > 0.5).astype(np.uint8)
        forged_pct = mask.mean() * 100

        # verdict
        if self.model:
            # تم التعديل هنا لربط النص بالـ Threshold
            if forged_pct >= FORGERY_THRESHOLD:
                vt, vc, vbg = f"⚠  FORGERY\nDETECTED\n{forged_pct:.1f}% flagged", C["red"],   "#1e0810"
            else:
                vt, vc, vbg = f"✓  AUTHENTIC\nIMAGE\n{forged_pct:.1f}% flagged",  C["green"], "#081a10"
        else:
            vt, vc, vbg = "MODEL NOT\nLOADED", C["warn"], C["card2"]

        self._vc.config(bg=vbg)
        self._vl.config(text=vt, fg=vc, bg=vbg)

        # matplotlib global style
        plt.rcParams.update({
            'figure.facecolor': C["bg"],
            'axes.facecolor':   C["card"],
            'axes.edgecolor':   C["border"],
            'text.color':       C["text"],
            'font.family':      'monospace',
        })

        self._render_images_tab(orig_pil, rgb, ela, noise, edge, mask, forged_pct)
        self._render_charts_tab(rgb, ela, noise, edge, mask, forged_pct)

        self._set_status("Analysis complete ✓", 100)
        self._btn.config(state="normal")
        self._tabs.show(self._tab_img)

    # ──────────────────────────────────────────────────────────────────
    #  TAB 1 — FORENSIC IMAGES
    # ──────────────────────────────────────────────────────────────────
    def _render_images_tab(self, orig_pil, rgb, ela, noise, edge, mask, forged_pct):
        tab = self._tabs.frame(self._tab_img)
        for w in tab.winfo_children(): w.destroy()

        # ── header inside tab ──────────────────────────────────────────
        hdr = tk.Frame(tab, bg=C["card"], height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  FORENSIC CHANNEL VIEWER",
                 fg=C["accent"], bg=C["card"], font=MFT).pack(side=tk.LEFT, padx=8, pady=6)
        tk.Label(hdr, text="Original · ELA · Noise · Edge · Mask · Overlay",
                 fg=C["text2"], bg=C["card"], font=MFL).pack(side=tk.LEFT)
        tk.Frame(tab, bg=C["border"], height=1).pack(fill=tk.X)

        # ── figure ────────────────────────────────────────────────────
        # Layout: 2 rows × 3 cols
        # Row 0: [Original full-res (col 0-1)] | [ELA] | [Noise]
        # Row 1: [Edge]                         | [Mask]| [Overlay]
        # Cyan dividers drawn via fig lines

        orig_arr = np.array(orig_pil.convert("RGB"))

        fig = plt.Figure(figsize=(13.8, 7.4), dpi=100, facecolor=C["bg"])
        gs  = mgridspec.GridSpec(
            2, 3, figure=fig,
            left=0.01, right=0.99, top=0.93, bottom=0.03,
            wspace=0.04, hspace=0.22
        )

        panels = [
            # (row, col, rowspan, colpan, data, cmap, title)
            (0, 0, 1, 1, orig_arr,        None,       "01  ORIGINAL IMAGE"),
            (0, 1, 1, 1, ela.squeeze(),   "magma",    "02  ERROR LEVEL ANALYSIS (ELA)"),
            (0, 2, 1, 1, noise.squeeze(), "inferno",  "03  NOISE RESIDUAL MAP"),
            (1, 0, 1, 1, edge.squeeze(),  "bone",     "04  CANNY EDGE FEATURES"),
            (1, 1, 1, 1, mask,            "hot",      "05  FORGERY PREDICTION MASK"),
            (1, 2, 1, 1, None,            None,       "06  OVERLAY  (mask on original)"),
        ]

        for (r, c, rs, cs, data, cmap, title) in panels:
            ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
            ax.set_facecolor(C["card"])
            ax.set_title(title, color=C["accent"], fontsize=8.5, pad=7,
                         fontfamily="monospace", fontweight="bold", loc="left", x=0.01)
            ax.axis('off')

            if data is None:            # overlay panel
                ax.imshow(rgb, interpolation="nearest")
                ov = np.zeros((*mask.shape, 4))
                # تم التعديل هنا لربط اللون بالـ Threshold
                if forged_pct >= FORGERY_THRESHOLD:
                    ov[mask==1] = [1.0, 0.24, 0.44, 0.55] # لون أحمر
                else:
                    ov[mask==1] = [0.22, 0.83, 0.33, 0.45] # لون أخضر
                ax.imshow(ov, interpolation="nearest")
            elif cmap:
                ax.imshow(data, cmap=cmap, interpolation="nearest")
            else:
                ax.imshow(data, interpolation="nearest")

            # subtle border per panel
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)

        # horizontal cyan divider line between row 0 and row 1
        # (drawn as figure line in axes-fraction space)
        fig.add_artist(
            plt.Line2D([0.005, 0.995], [0.505, 0.505],
                       transform=fig.transFigure,
                       color=C["div"], linewidth=1.2, alpha=0.6)
        )
        # vertical cyan dividers between columns
        for xpos in [0.343, 0.668]:
            fig.add_artist(
                plt.Line2D([xpos, xpos], [0.03, 0.93],
                           transform=fig.transFigure,
                           color=C["div"], linewidth=1.0, alpha=0.5)
            )

        fig.suptitle("MUST AI FORENSICS  ·  CHANNEL ANALYSIS",
                     color=C["text2"], fontsize=8, fontfamily="monospace", y=0.975)

        self._fig1 = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.configure(bg=C["bg"], highlightthickness=0)
        cw.pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────────────────────────────
    #  TAB 2 — ANALYSIS CHARTS
    # ──────────────────────────────────────────────────────────────────
    def _render_charts_tab(self, rgb, ela, noise, edge, mask, forged_pct):
        tab = self._tabs.frame(self._tab_ch)
        for w in tab.winfo_children(): w.destroy()

        hdr = tk.Frame(tab, bg=C["card"], height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  STATISTICAL ANALYSIS DASHBOARD",
                 fg=C["accent"], bg=C["card"], font=MFT).pack(side=tk.LEFT, padx=8, pady=6)
        tk.Label(hdr, text="Pixel distribution · Intensity histograms · Channel comparison",
                 fg=C["text2"], bg=C["card"], font=MFL).pack(side=tk.LEFT)
        tk.Frame(tab, bg=C["border"], height=1).pack(fill=tk.X)

        fig = plt.Figure(figsize=(13.8, 7.4), dpi=100, facecolor=C["bg"])
        gs  = mgridspec.GridSpec(
            2, 3, figure=fig,
            left=0.06, right=0.97, top=0.91, bottom=0.09,
            wspace=0.35, hspace=0.42
        )

        ax_style = dict(facecolor=C["card2"])

        # ── Chart 1: Authentic vs Forged donut ────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(C["card2"])
        clean = 100 - forged_pct
        wedge_colors = [C["green"], C["red"]]
        wedges, texts, autotexts = ax1.pie(
            [clean, forged_pct],
            labels=["Authentic", "Flagged"],
            colors=wedge_colors,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor=C["bg"], linewidth=2)
        )
        for t in texts:      t.set(color=C["text3"], fontsize=7.5, fontfamily="monospace")
        for t in autotexts:  t.set(color=C["bg"],    fontsize=7.5, fontfamily="monospace", fontweight="bold")
        ax1.set_title("PIXEL VERDICT DISTRIBUTION", color=C["accent"],
                      fontsize=8.5, pad=10, fontfamily="monospace", fontweight="bold")

        # ── Chart 2: RGB channel histograms ───────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(C["card2"])
        ch_colors = ["#ff5370", "#39d353", "#00e5ff"]
        ch_labels = ["Red", "Green", "Blue"]
        for i, (col, lbl) in enumerate(zip(ch_colors, ch_labels)):
            vals = rgb[:, :, i].flatten()
            ax2.hist(vals, bins=64, color=col, alpha=0.72, label=lbl,
                     histtype='stepfilled', linewidth=0)
        ax2.set_title("RGB CHANNEL DISTRIBUTION", color=C["accent"],
                      fontsize=8.5, pad=6, fontfamily="monospace", fontweight="bold")
        ax2.tick_params(colors=C["text3"], labelsize=6.5)
        ax2.set_xlabel("Pixel intensity", color=C["text2"], fontsize=7)
        ax2.set_ylabel("Frequency", color=C["text2"], fontsize=7)
        ax2.legend(fontsize=6.5, facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"])
        for sp in ax2.spines.values():
            sp.set_edgecolor(C["border"])

        # ── Chart 3: ELA intensity histogram ──────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(C["card2"])
        ela_vals = ela.squeeze().flatten()
        ax3.hist(ela_vals, bins=64, color=C["warn"], alpha=0.85,
                 histtype='stepfilled', linewidth=0)
        ax3.axvline(ela_vals.mean(), color=C["red"], linestyle="--",
                    linewidth=1.2, label=f"Mean: {ela_vals.mean():.3f}")
        ax3.set_title("ELA INTENSITY HISTOGRAM", color=C["accent"],
                      fontsize=8.5, pad=6, fontfamily="monospace", fontweight="bold")
        ax3.tick_params(colors=C["text3"], labelsize=6.5)
        ax3.set_xlabel("ELA value", color=C["text2"], fontsize=7)
        ax3.set_ylabel("Frequency", color=C["text2"], fontsize=7)
        ax3.legend(fontsize=6.5, facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"])
        for sp in ax3.spines.values():
            sp.set_edgecolor(C["border"])

        # ── Chart 4: Channel mean comparison bar ──────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(C["card2"])
        ch_means = [
            rgb[:,:,0].mean(), rgb[:,:,1].mean(), rgb[:,:,2].mean(),
            ela.squeeze().mean(), noise.squeeze().mean(), edge.squeeze().mean()
        ]
        ch_names = ["R", "G", "B", "ELA", "Noise", "Edge"]
        bar_cols  = [C["red"], C["green"], C["accent"], C["warn"], C["purple"], C["text3"]]
        bars = ax4.bar(ch_names, ch_means, color=bar_cols, edgecolor=C["bg"], linewidth=0.8)
        ax4.set_title("CHANNEL MEAN INTENSITY", color=C["accent"],
                      fontsize=8.5, pad=6, fontfamily="monospace", fontweight="bold")
        ax4.set_ylim(0, 1.0)
        ax4.tick_params(colors=C["text3"], labelsize=7)
        ax4.set_ylabel("Mean (0–1)", color=C["text2"], fontsize=7)
        for b, v in zip(bars, ch_means):
            ax4.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}",
                     ha="center", va="bottom", color=C["text"], fontsize=6.5,
                     fontfamily="monospace")
        for sp in ax4.spines.values():
            sp.set_edgecolor(C["border"])

        # ── Chart 5: Noise residual histogram ─────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(C["card2"])
        nv = noise.squeeze().flatten()
        ax5.hist(nv, bins=64, color=C["purple"], alpha=0.85,
                 histtype='stepfilled', linewidth=0)
        ax5.axvline(nv.mean(), color=C["accent"], linestyle="--",
                    linewidth=1.2, label=f"Mean: {nv.mean():.3f}")
        ax5.axvline(nv.std(),  color=C["warn"],   linestyle=":",
                    linewidth=1.0, label=f"Std:  {nv.std():.3f}")
        ax5.set_title("NOISE RESIDUAL HISTOGRAM", color=C["accent"],
                      fontsize=8.5, pad=6, fontfamily="monospace", fontweight="bold")
        ax5.tick_params(colors=C["text3"], labelsize=6.5)
        ax5.set_xlabel("Noise value", color=C["text2"], fontsize=7)
        ax5.set_ylabel("Frequency", color=C["text2"], fontsize=7)
        ax5.legend(fontsize=6.5, facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"])
        for sp in ax5.spines.values():
            sp.set_edgecolor(C["border"])

        # ── Chart 6: Row-wise forgery density ─────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(C["card2"])
        row_density = mask.mean(axis=1)    # average flagged pixels per row
        ax6.fill_between(range(len(row_density)), row_density,
                         color=C["red"], alpha=0.55)
        ax6.plot(row_density, color=C["red"], linewidth=1.0)
        ax6.set_title("FORGERY DENSITY  (row-wise)", color=C["accent"],
                      fontsize=8.5, pad=6, fontfamily="monospace", fontweight="bold")
        ax6.tick_params(colors=C["text3"], labelsize=6.5)
        ax6.set_xlabel("Image row (top → bottom)", color=C["text2"], fontsize=7)
        ax6.set_ylabel("Flagged ratio", color=C["text2"], fontsize=7)
        ax6.set_ylim(0, 1)
        for sp in ax6.spines.values():
            sp.set_edgecolor(C["border"])

        # horizontal cyan divider
        fig.add_artist(
            plt.Line2D([0.01, 0.99], [0.50, 0.50],
                       transform=fig.transFigure,
                       color=C["div"], linewidth=1.0, alpha=0.45)
        )
        # vertical cyan dividers
        for xp in [0.36, 0.675]:
            fig.add_artist(
                plt.Line2D([xp, xp], [0.06, 0.92],
                           transform=fig.transFigure,
                           color=C["div"], linewidth=0.8, alpha=0.4)
            )

        fig.suptitle("MUST AI FORENSICS  ·  STATISTICAL ANALYSIS",
                     color=C["text2"], fontsize=8, fontfamily="monospace", y=0.965)

        self._fig2 = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        cw = canvas.get_tk_widget()
        cw.configure(bg=C["bg"], highlightthickness=0)
        cw.pack(fill=tk.BOTH, expand=True)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    ForgeryDetectorApp(root)
    root.mainloop()