"""Read-only result views for the E2M3 -> E2M1 mantissa experiment.

No tensor or dataframe dependency is required. Matplotlib is optional. The only
writes are an explicitly generated notebook/helper and new figures under the
chosen run output directory; result CSV/JSON files are never changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


NOTEBOOK_NAME = "Mantissa_MSB_exponent_probe_resultaten.ipynb"
NUMERIC_SCHEMA = "mx-e2m3-e2m1-mantissa-rules-numeric-summary-v1"
METHODS = ("lsb_trunc", "rne", "msb_bestexp")
VARIANTS = ("source8", "source6", *METHODS)
LABELS = {
    "source8": "E4M3-broncontrole",
    "source6": "E2M3-broncontrole",
    "lsb_trunc": "LSB-truncatie",
    "rne": "Afronden (RNE)",
    "msb_bestexp": "MSB's weg + beste exponent",
}
BITS = {"source8": 8, "source6": 6, **{method: 4 for method in METHODS}}
COLORS = {"lsb_trunc": "#326A98", "rne": "#268577", "msb_bestexp": "#B96339"}


def _json(path):
    def reject_constant(value):
        raise ValueError("Niet-eindige JSON-waarde: " + value)
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON-root is geen object")
    return value


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Niet-eindig getal")
    return value


def _integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean waar een geheel getal verwacht werd")
    result = int(value)
    if float(value) != result:
        raise ValueError("Geen geheel getal: " + str(value))
    return result


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(_cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(_cell, row)) + " |" for row in rows),
    ])


def _numeric(summary, rows):
    if summary.get("schema") != NUMERIC_SCHEMA:
        raise ValueError("Onbekend numeric_summary-schema")
    assumptions = summary["assumptions"]
    if (assumptions.get("shared_scale") != 1
            or assumptions.get("shared_scale_unchanged") is not True
            or assumptions.get("population_is_model_weight_distribution") is not False):
        raise ValueError("De numerieke analyse bevestigt uniforme codeweging bij S=1 niet")
    if set(summary["methods"]) != set(METHODS) or len(rows) != 192:
        raise ValueError("Verwacht drie methoden en precies 192 codetabelrijen")
    parsed = []
    for raw in rows:
        row = dict(raw)
        for key in ("source_code", "source_sign", "source_exponent", "source_mantissa",
                    "target_code", "target_sign", "target_exponent", "target_mantissa"):
            row[key] = _integer(row[key])
        for key in ("source_value", "target_value", "error", "abs_error", "squared_error"):
            row[key] = _number(row[key])
        if row["method"] not in METHODS or not 0 <= row["source_code"] < 64 or not 0 <= row["target_code"] < 16:
            raise ValueError("Onverwachte methode of code in de codetabel")
        if row["source_sign"] != row["source_code"] >> 5:
            raise ValueError("Bronteken en broncode zijn inconsistent")
        delta = row["target_value"] - row["source_value"]
        if not (row["error"] == delta and row["abs_error"] == abs(delta)
                and row["squared_error"] == delta * delta):
            raise ValueError("Foutkolommen stemmen niet overeen met bron- en doelwaarden")
        parsed.append(row)
    if {(r["source_code"], r["method"]) for r in parsed} != {(c, m) for c in range(64) for m in METHODS}:
        raise ValueError("Codetabel mist codes of bevat duplicaten")
    for population, codes in {
        "all64": set(range(64)), "positive32": set(range(32)),
        "positive_nonzero31": set(range(1, 32)), "positive_normal24": set(range(8, 32)),
    }.items():
        data = summary["populations"][population]
        if data["count"] != len(codes):
            raise ValueError("Onjuist aantal codes in " + population)
        for method in METHODS:
            subset = [r for r in parsed if r["method"] == method and r["source_code"] in codes]
            expected = {"mae": sum(r["abs_error"] for r in subset) / len(codes),
                        "mse": sum(r["squared_error"] for r in subset) / len(codes),
                        "max_abs_error": max(r["abs_error"] for r in subset)}
            for metric, value in expected.items():
                if not math.isclose(_number(data["methods"][method][metric]), value, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"CSV en samenvatting verschillen voor {population}/{method}/{metric}")
    return parsed


def _accuracy(rows):
    if len(rows) != len(VARIANTS) or {r["variant"] for r in rows} != set(VARIANTS):
        raise ValueError("Verwacht precies de vijf volledige netwerkvarianten")
    parsed = []
    for raw in rows:
        row = dict(raw)
        for key in ("weight_bits", "n", "correct1", "correct5"):
            row[key] = _integer(row[key])
        for key in ("top1_percent", "top5_percent", "loss"):
            row[key] = _number(row[key])
        if row["weight_bits"] != BITS[row["variant"]]:
            raise ValueError("Variant en gewichtsbreedte verschillen")
        if not 0 <= row["correct1"] <= row["correct5"] <= row["n"] or row["n"] <= 0:
            raise ValueError("Ongeldige aantallen correct/n")
        for correct, percent in (("correct1", "top1_percent"), ("correct5", "top5_percent")):
            if not math.isclose(row[percent], 100 * row[correct] / row["n"], rel_tol=0, abs_tol=1e-6):
                raise ValueError("Percentage en exact aantal verschillen")
        parsed.append(row)
    if len({r["n"] for r in parsed}) != 1:
        raise ValueError("Varianten hebben verschillende aantallen voorbeelden")
    return sorted(parsed, key=lambda row: VARIANTS.index(row["variant"]))


def read_results(out: Path) -> dict:
    """Read observed results; failures become visible diagnostics, never PASS."""
    out = Path(out).expanduser().resolve()
    result = {"directory": out, "report": {}, "reference": {}, "numeric": None,
              "mapping": [], "accuracy": [], "weight_error": [], "errors": [], "notes": []}
    reference_name = "reference.json" if (out / "reference.json").is_file() else "REFERENCE.json"
    for name, key in (("run_report.json", "report"), (reference_name, "reference")):
        path = out / name
        if path.is_file():
            try:
                result[key] = _json(path)
            except (OSError, ValueError, TypeError) as exc:
                result["errors"].append(f"{name}: {exc}")
        elif key == "report":
            result["notes"].append("run_report.json ontbreekt; voltooiing en runstatus zijn niet vastgesteld.")
    if not result["reference"] and isinstance(result["report"].get("reference"), dict):
        result["reference"] = result["report"]["reference"]
    if (out / "numeric_summary.json").is_file() and (out / "codebook_mapping.csv").is_file():
        try:
            summary = _json(out / "numeric_summary.json")
            mapping = _numeric(summary, _csv(out / "codebook_mapping.csv"))
            result.update(numeric=summary, mapping=mapping)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["errors"].append("Numerieke resultaten: " + str(exc))
    else:
        result["notes"].append("De volledige numerieke resultaten zijn nog niet beschikbaar.")
    network = result["report"].get("network_evaluated")
    accuracy_path = out / "accuracy.csv"
    if network is True and accuracy_path.is_file():
        try:
            result["accuracy"] = _accuracy(_csv(accuracy_path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["errors"].append("Netwerkresultaten: " + str(exc))
    elif network is True:
        result["errors"].append("run_report meldt een netwerkevaluatie, maar accuracy.csv ontbreekt.")
    elif accuracy_path.is_file():
        result["errors"].append("accuracy.csv bestaat, maar run_report bevestigt geen netwerkevaluatie; accuracy is niet weergegeven.")
    else:
        result["notes"].append("Geen netwerkevaluatie beschikbaar; er wordt geen accuracy getoond.")
    if (out / "weight_error.csv").is_file():
        try:
            result["weight_error"] = _csv(out / "weight_error.csv")
        except (OSError, ValueError, TypeError) as exc:
            result["errors"].append("weight_error.csv: " + str(exc))
    return result


def status_markdown(result):
    report = result["report"]
    raw_status = str(report.get("status", "onbekend"))
    failed = raw_status.upper() in {"ERROR", "FAIL", "FAILED", "FOUT"} or bool(result["errors"])
    heading = "FOUT — beoordeel onderstaande meldingen" if failed else "Runstatus: " + raw_status
    synthetic = report.get("synthetic_only")
    scope = ("Synthetische technische controle; geen onderzoeksaccuracy." if synthetic is True else
             "Niet-synthetische uitvoering volgens het runrapport." if synthetic is False else
             "Synthetische of echte uitvoering: niet vastgesteld.")
    rows = [["Commando", report.get("command", "onbekend")],
            ["Netwerkevaluatie bevestigd", "ja" if report.get("network_evaluated") is True else "nee"],
            ["Numerieke tabel gecontroleerd", "ja; 192 CSV-rijen" if result["numeric"] else "nee"],
            ["Gemeten netwerkvarianten", len(result["accuracy"])]]
    lines = ["## " + heading, scope, _table(["Onderdeel", "Waarneming"], rows)]
    messages = list(result["errors"])
    for key in ("error", "errors", "exception", "warnings"):
        value = report.get(key)
        if value:
            messages.extend([f"{key}: {item}" for item in value] if isinstance(value, list) else [f"{key}: {value}"])
    messages.extend(result["notes"])
    if messages:
        lines.extend(["### Meldingen", "\n".join("- " + _cell(message) for message in messages)])
    if report.get("protocol"):
        protocol = report["protocol"]
        text = json.dumps(protocol, ensure_ascii=False, indent=2) if not isinstance(protocol, str) else protocol
        lines.extend(["### Uitgevoerd protocol", "```text\n" + text + "\n```"])
    return "\n\n".join(lines)


def numeric_markdown(result):
    summary = result["numeric"]
    if not summary:
        return "Geen volledige, consistente numerieke tabel beschikbaar. Zie de runstatus."
    population = summary["populations"]["positive32"]
    rows = []
    for method in METHODS:
        data = population["methods"][method]
        rows.append([LABELS[method], f'{data["mae"]:.6f}', f'{data["mse"]:.8f}',
                     f'{data["max_abs_error"]:.6f}', f'{data["magnitude_bias"]:.6f}',
                     f'{data["exact_count"]}/{population["count"]}'])
    text = ["## Numerieke fout bij uniforme codeweging, S = 1",
            "Alle 32 codes met positief teken, inclusief +0, krijgen hetzelfde gewicht. "
            "Dit is geen gemeten gewichtsverdeling van het netwerk. MAE is de gemiddelde absolute fout; "
            "MSE is de gemiddelde gekwadrateerde fout. De bias hieronder is het gemiddelde van |doel| − |bron|.",
            _table(["Methode", "MAE", "MSE", "Max. absolute fout", "Magnitude-bias", "Exact"], rows)]
    comparisons = []
    for method in METHODS:
        data = population["methods"][method]
        comparisons.append([LABELS[method], data["better_than_lsb"], data["equal_to_lsb"],
                            data["worse_than_lsb"], summary["monotonicity"][method]["decreases"]])
    text.extend(["### Vergelijking per broncode",
                 _table(["Methode", "Kleinere fout dan LSB", "Gelijke fout", "Grotere fout", "Dalingen in mapping"], comparisons),
                 "Een daling betekent dat een grotere positieve bronwaarde een kleinere doelwaarde krijgt; "
                 "er zijn 31 opeenvolgende overgangen tussen de 32 broncodes."])
    extra = []
    for population_name, label in (("all64", "Alle 64 codes, beide tekens"),
                                   ("positive_nonzero31", "31 positieve niet-nulcodes"),
                                   ("positive_normal24", "24 positieve normale codes")):
        population = summary["populations"][population_name]
        for method in METHODS:
            data = population["methods"][method]
            extra.append([label, LABELS[method], population["count"], f'{data["mae"]:.6f}', f'{data["mse"]:.8f}'])
    text.extend(["### Expliciete codepopulaties",
                 _table(["Populatie", "Methode", "Aantal", "MAE", "MSE"], extra),
                 "De 64-codepopulatie telt +0 en −0 als twee encoderingen. De exacte breuken staan in "
                 "numeric_summary.json; de 192 afzonderlijke beslissingen staan in codebook_mapping.csv."])
    checks = summary.get("exhaustive_checks", {})
    failures = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    verified = checks.get("codes_checked") == 64 and not failures
    closed = all(row["target_code"] == ((row["source_code"] >> 5) << 3)
                 | (min(3, ((row["source_code"] >> 3) & 3) + int((row["source_code"] & 7) == 6)) << 1)
                 | (row["source_code"] & 1)
                 for row in result["mapping"] if row["method"] == "msb_bestexp")
    text.extend(["### Controle van de regels",
                 f"Uitputtende checks over 64 codes: {'bevestigd' if verified else 'NIET bevestigd'}. "
                 f"Compacte exponentregel gelijk aan de opgeslagen zoekresultaten: {'ja' if closed else 'NEE'}."])
    if failures:
        text.append("Niet geslaagde checks: " + ", ".join(failures))
    return "\n\n".join(text)


def accuracy_markdown(result):
    rows = result["accuracy"]
    if not rows:
        return "## Netwerkaccuracy\n\nGeen volledige, bevestigde netwerkevaluatie beschikbaar. " \
               "De numerieke codetabel levert geen accuracywaarden."
    synthetic = result["report"].get("synthetic_only") is True
    title = "## Synthetische technische netwerkcontrole" if synthetic else "## Gemeten netwerkaccuracy"
    text = [title]
    if synthetic:
        text.append("Deze waarden horen bij een synthetische fixture en zijn geen SVHN-onderzoeksresultaat.")
    text.append(_table(["Variant", "Gewichtsbits", "Correct top-1 / n", "Top-1 (%)", "Top-5 (%)", "Verlies"],
                       [[LABELS[row["variant"]], row["weight_bits"], f'{row["correct1"]}/{row["n"]}',
                         f'{row["top1_percent"]:.6f}', f'{row["top5_percent"]:.6f}', f'{row["loss"]:.6f}']
                        for row in rows]))
    baseline = next(row for row in rows if row["variant"] == "lsb_trunc")
    text.extend(["### Vierbitsvergelijking op dezelfde voorbeelden",
                 _table(["Methode", "Verschil correcte voorspellingen versus LSB", "Verschil top-1 (procentpunt)"],
                        [[LABELS[row["variant"]], f'{row["correct1"] - baseline["correct1"]:+d}',
                          f'{100 * (row["correct1"] - baseline["correct1"]) / row["n"]:+.6f}']
                         for row in rows if row["weight_bits"] == 4])])
    expected = result["reference"].get("expected_metrics", {})
    if expected and not synthetic:
        comparisons = []
        for variant in ("source8", "source6", "lsb_trunc"):
            row = next(row for row in rows if row["variant"] == variant)
            ref = expected.get(str(row["weight_bits"]), {})
            if all(key in ref for key in ("correct1", "n")):
                comparisons.append([LABELS[variant], f'{row["correct1"]}/{row["n"]}',
                                    f'{ref["correct1"]}/{ref["n"]}', row["correct1"] - ref["correct1"]])
        text.extend(["### Reproductiecontrole tegenover de bewaarde epoch100-referentie",
                     _table(["Controle", "Nu gemeten", "Eerdere referentie", "Verschil correct"], comparisons),
                     "De referentiewaarden zijn eerder gemeten controles, geen nieuwe metingen. "
                     "Een afwijking vraagt beoordeling van het runrapport en de uitvoeromgeving."])
    text.append("De vergelijking is beschrijvend voor dit checkpoint en deze testset. "
                "Er zijn geen onafhankelijke trainingsherhalingen of statistische significantietoetsen uitgevoerd.")
    return "\n\n".join(text)


def weight_error_markdown(result):
    rows = result["weight_error"]
    if not rows:
        return "## Fout op de werkelijke gewichten\n\nGeen afzonderlijke gewichtsdiagnostiek beschikbaar. " \
               "De uniforme codetabelfout mag niet als gemeten netwerkgewichts-MAE of -MSE worden gelezen."
    required = {"layer", "method", "n_weights", "mae_vs_e2m3", "mse_vs_e2m3", "max_abs_error",
                "zero_target", "exponent_fields_changed", "padding_excluded", "shared_scale_unchanged"}
    if not required <= set(rows[0]):
        return "## Fout op de werkelijke gewichten\n\nweight_error.csv bestaat, maar het schema " \
               "wordt door deze notebook niet herkend. Bekijk het bestand en het uitvoerrapport."
    try:
        for row in rows:
            if row["method"] not in METHODS or _integer(row["n_weights"]) <= 0:
                raise ValueError("Onbekende methode of ongeldige gewichtsnoemer")
            if any(str(row[key]).lower() != "true" for key in ("padding_excluded", "shared_scale_unchanged")):
                raise ValueError("Uitsluiting van padding of behoud van de schaal niet bevestigd")
        def metric(row, key):
            return f"{_number(row[key]):.6g}"
        totals = [row for row in rows if row["layer"] == "ALL_MX_LAYERS"]
        details = [row for row in rows if row["layer"] != "ALL_MX_LAYERS"]
        aggregate = [[LABELS[row["method"]], _integer(row["n_weights"]), metric(row, "mae_vs_e2m3"),
                      metric(row, "mse_vs_e2m3"), metric(row, "max_abs_error"),
                      _integer(row["zero_target"]), _integer(row["exponent_fields_changed"])] for row in totals]
        layers = [[row["layer"], LABELS[row["method"]], _integer(row["n_weights"]),
                   metric(row, "mae_vs_e2m3"), metric(row, "mse_vs_e2m3")] for row in details]
    except (KeyError, ValueError, TypeError) as exc:
        return "## FOUT in de gewichtsdiagnostiek\n\n" + _cell(exc)
    text = ["## Fout op de werkelijke gewichten",
            "Referentie is de gedecodeerde E2M3-bron, inclusief de opgeslagen schaal per blok. "
            "Iedere werkelijke tensorwaarde telt één keer; padding telt niet mee en de gedeelde schaal "
            "blijft gelijk. De totalen wegen de zes MX-lagen naar hun aantal gewichten.",
            _table(["Methode", "Gewichten", "MAE vs E2M3", "MSE vs E2M3", "Max. absolute fout",
                    "Nulwaarden doel", "Exponentvelden gewijzigd"], aggregate),
            "### Per MX-laag",
            _table(["Laag", "Methode", "Gewichten", "MAE vs E2M3", "MSE vs E2M3"], layers),
            "De volledige diagnostiek, waaronder RMSE en genormaliseerde MSE, staat in weight_error.csv. "
            "Deze fout gebruikt de werkelijke gewichtsverdeling en mag niet worden verward met de uniforme codetabelfout."]
    return "\n\n".join(text)


def provenance_markdown(result):
    reference = result["reference"]
    if not reference:
        return "## Herkomst\n\nHet referentiebestand is niet beschikbaar in deze resultaatmap. " \
               "De actuele broncontroles moeten uit run_report.json worden beoordeeld."
    rows = [["Verwachte bron", reference.get("source_file", "onbekend")],
            ["Verwachte epoch", reference.get("source_epoch", "onbekend")],
            ["Verwachte bron-SHA256", reference.get("source_sha256", "onbekend")],
            ["Verwachte checkpoint-SHA256", reference.get("checkpoint_sha256", "onbekend")]]
    return "## Herkomst\n\n" + _table(["Referentiegegeven", "Waarde"], rows) + \
           "\n\nDit zijn de bedoelde brongegevens uit het referentiebestand (reference.json, een kopie van pakket-REFERENCE.json). Of een bron werkelijk is geladen " \
           "en geverifieerd, staat in het uitvoerrapport; een numerieke analyse alleen laadt geen netwerk."


def plot_numeric(result):
    if not result["numeric"]:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.7), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        rows = sorted((r for r in result["mapping"] if r["method"] == method and r["source_sign"] == 0),
                      key=lambda row: row["source_code"])
        axis.plot([0, 7.5], [0, 7.5], "--", color="#737B84", linewidth=1.2, label="Ideaal: doel = bron")
        axis.plot([r["source_value"] for r in rows], [r["target_value"] for r in rows],
                  "o-", color=COLORS[method], markersize=4.1, linewidth=1.2, label="Codemapping")
        axis.set(title=LABELS[method], xlabel="E2M3-bronwaarde (S = 1)", xlim=(-.15, 7.8), ylim=(-.15, 7.8))
        axis.grid(alpha=.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("E2M1-doelwaarde (S = 1)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("E2M3 → E2M1 bij vaste gedeelde schaal S = 1", fontsize=14, y=.98)
    fig.text(.5, .025, "32 broncodes met positief teken, inclusief +0; uniforme codeweging, geen netwerkgewichtsverdeling.",
             ha="center", fontsize=9, color="#4A535D")
    fig.tight_layout(rect=(0, .065, 1, .93))
    return fig


def plot_accuracy(result):
    if not result["accuracy"]:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.1, 2.1]})
    for axis, bit in zip(axes, (8, 6, 4)):
        rows = [row for row in result["accuracy"] if row["weight_bits"] == bit]
        for index, row in enumerate(rows):
            percent = row["top1_percent"]
            axis.scatter(percent, index, s=63, marker="o", color=COLORS.get(row["variant"], "#657789"), zorder=3)
            left = percent > 96
            axis.annotate(f"{percent:.4f}%", (percent, index), xytext=(-8 if left else 8, 0),
                          textcoords="offset points", va="center", ha="right" if left else "left", fontsize=10)
        axis.set_yticks(range(len(rows)), [LABELS[row["variant"]] for row in rows])
        axis.set_ylim(len(rows) - .45, -.55)
        axis.set_xlim(0, 100)
        axis.set_title(f"{bit} gewichtsbits — " + ("E2M1-varianten; vaste BN-bank 4" if bit == 4 else "broncontrole"),
                       loc="left", fontsize=11)
        axis.grid(axis="x", alpha=.22)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Top-1-accuracy (%) — dezelfde 0–100%-as in ieder paneel")
    title = "Synthetische technische netwerkcontrole" if result["report"].get("synthetic_only") is True else "Gemeten netwerkaccuracy per precisie"
    fig.suptitle(title, fontsize=14, y=.98)
    fig.text(.5, .015, f"n = {result['accuracy'][0]['n']}; gelijke cirkelgrootte. Geen foutbalken: één checkpoint en één evaluatieset.",
             ha="center", fontsize=9, color="#4A535D")
    fig.tight_layout(rect=(0, .045, 1, .95))
    return fig


def create_figures(out: Path) -> list[Path]:
    """Add PNG figures to a run output directory; never replace existing files."""
    out = Path(out).expanduser().resolve()
    if not out.is_dir():
        raise FileNotFoundError("De uitvoermap bestaat niet: " + str(out))
    result = read_results(out)
    paths = []
    for filename, make in (("numeric_code_mapping.png", plot_numeric), ("accuracy_by_precision.png", plot_accuracy)):
        figure = make(result)
        if figure is None:
            continue
        folder = out / "figures"
        folder.mkdir(exist_ok=True)
        if folder.is_symlink():
            raise ValueError("De figurenmap moet binnen de nieuwe uitvoermap liggen")
        path = folder / filename
        try:
            with path.open("xb") as handle:
                figure.savefig(handle, format="png", dpi=180, facecolor="white")
        except FileExistsError:
            if not path.is_file() or path.is_symlink():
                raise ValueError("Bestaand figuurpad is geen regulier uitvoerbestand")
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)
        paths.append(path)
    return paths


SCOPE = r"""## Wat deze proef onderzoekt

De bron is E2M3; het doel blijft standaard E2M1 met hetzelfde teken en dezelfde
gedeelde E8M0-schaal. Schrijf de drie fractionele mantissabits als $m_2m_1m_0$.

| Methode | Bewaarde informatie en keuze |
| --- | --- |
| LSB-truncatie | Verwijder $m_1m_0$; behoud $m_2$ en de exponent. |
| RNE | Kies de dichtstbijzijnde E2M1-waarde; bij gelijke afstand de even doelcode. |
| MSB's weg + beste exponent | Verwijder $m_2m_1$; behoud $m_0$ en zoek de beste van vier exponentcodes. |

**Beste exponent** betekent hier de kleinste absolute fout onder de beperking
$m'=m_0$, bij vast teken en vaste schaal. Bij gelijke fout gaat de voorkeur naar
de kleinste exponentwijziging en daarna naar de laagste exponentcode. Het is geen
optimalisatie van netwerkaccuracy en geen verliesvrije overdracht van informatie.

Voor deze volledige codetabel reduceert de zoekprocedure tot de volgende regel,
met $E\in\{0,1,2,3\}$ de **opgeslagen exponentcode** en $m\in\{0,\ldots,7\}$:

$$m'=m\bmod2,\qquad
E'=\begin{cases}\min(E+1,3),&m=6,\\E,&m\ne6.\end{cases}$$

De resultaatcontrole vergelijkt die compacte regel met alle 64 zoekresultaten.
Deze regel kan dus ook de verwijderde bits raadplegen voordat ze worden weggegooid.
De proef meet geen siliciumoppervlakte, energieverbruik of geheugentoegangskosten.

RNE heeft bij deze vaste schaal en hetzelfde doelcodeboek de kleinste absolute
en gekwadrateerde fout per element. Dat voorspelt de netwerkaccuracy niet volledig.
"""


LIMITS = """## Interpretatie van een eventuele netwerkevaluatie

De netwerkproef gebruikt één bevroren bron uit epoch 100. Alle drie
vierbitsvarianten gebruiken dezelfde opgeslagen E2-schalen en dezelfde BN-bank 4,
die tijdens de training met LSB-truncatie is geleerd. De E4M3- en E2M3-controles
gebruiken respectievelijk hun eigen banken 8 en 32. Activaties blijven A32.

Er wordt niets getraind en er vindt geen BN-herkalibratie plaats. Een verschil
beschrijft daarom het effect van deze posthocvervanging binnen dit checkpoint.
Een slechter resultaat bewijst niet dat dezelfde representatie na aangepaste
quantization-aware training slecht moet presteren. Een beter resultaat bewijst
evenmin een algemene accuracywinst, een optimale trainingsmethode of hardwarewinst.

De numerieke tabellen wegen codes uniform. Gewichtsdiagnostiek, als beschikbaar,
weegt de werkelijk aanwezige tensorwaarden. Testaccuracy telt correcte voorspellingen.
Deze drie grootheden beantwoorden verschillende vragen.
"""


def _new_text(path, text):
    data = text.encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FileExistsError("Bestaand afwijkend bestand blijft ongewijzigd: " + str(path))


def write_notebook(out: Path) -> Path:
    """Create the standalone result notebook and copy this helper alongside it."""
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve()
    _new_text(out / "probe_notebook.py", helper.read_text(encoding="utf-8"))
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
        item["id"] = hashlib.sha256((kind + "\n" + source).encode()).hexdigest()[:12]
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", "# E2M3 → E2M1: mantissa-MSB's verwijderen en exponent aanpassen\n\n"
         "Deze notebook leest de bestanden van één uitvoering. Numerieke coderegels en "
         "gemeten netwerkresultaten worden afzonderlijk getoond. Runstatus en ontbrekende gegevens "
         "zijn zichtbaar; eerder gemeten referentieaccuracy wordt nooit als nieuwe meting ingevuld.")
    cell("code", "from pathlib import Path\nimport sys\nfrom IPython.display import display, Markdown\n\n"
         "# Alleen aanpassen als de resultaatmap is verplaatst.\nRUN_DIR = None\n"
         f"_GENERATED_OUTPUT = Path({str(out)!r})\n"
         "_candidates = [Path(RUN_DIR).expanduser()] if RUN_DIR is not None else [Path.cwd(), _GENERATED_OUTPUT]\n"
         "OUT = next((p.resolve() for p in _candidates if (p / 'probe_notebook.py').is_file()\n"
         "            and any((p / name).is_file() for name in ('run_report.json', 'numeric_summary.json'))), None)\n"
         "if OUT is None:\n    raise FileNotFoundError('Stel RUN_DIR in op de map met run_report.json en probe_notebook.py.')\n"
         "sys.path.insert(0, str(OUT))\nimport probe_notebook as view\n"
         "if Path(view.__file__).resolve() != OUT / 'probe_notebook.py':\n"
         "    raise RuntimeError('Een helper uit een andere uitvoermap is al geïmporteerd; herstart de kernel.')\n"
         "results = view.read_results(OUT)\ndisplay(Markdown(view.status_markdown(results)))")
    cell("markdown", SCOPE)
    cell("code", "display(Markdown(view.numeric_markdown(results)))")
    cell("code", "figure = view.plot_numeric(results)\nif figure is not None:\n"
         "    display(figure)\n    import matplotlib.pyplot as plt\n    plt.close(figure)\nelse:\n"
         "    display(Markdown('Geen numerieke figuur: volledige geldige data of optioneel matplotlib ontbreekt.'))")
    cell("code", "display(Markdown(view.accuracy_markdown(results)))\nfigure = view.plot_accuracy(results)\n"
         "if figure is not None:\n    display(figure)\n    import matplotlib.pyplot as plt\n    plt.close(figure)")
    cell("code", "display(Markdown(view.weight_error_markdown(results)))")
    cell("markdown", LIMITS)
    cell("code", "display(Markdown(view.provenance_markdown(results)))")
    cell("markdown", "## Optionele uitvoerfiguren\n\n"
         "De runner kan PNG-figuren in de submap `figures` plaatsen. De onderstaande optionele stap "
         "maakt alleen ontbrekende figuren in deze resultaatmap; CSV- en JSON-invoer blijven ongewijzigd.")
    cell("code", "SAVE_FIGURES = False\nif SAVE_FIGURES:\n"
         "    for path in view.create_figures(OUT):\n        display(Markdown(f'[{path.name}](figures/{path.name})'))")
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}
    path = out / NOTEBOOK_NAME
    _new_text(path, json.dumps(notebook, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path
