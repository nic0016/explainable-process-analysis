# XES->DataFrame Encoding: Spezifikation und Beispiel

## Zielsetzung
Diese Dokumentation beschreibt das in diesem Projekt verwendete Encoding von XES-Eventlogs in maschinenlern-taugliche Repräsentationen. Neben dem Event-ID-basierten Sequenz-Encoding (One-Hot je Eventposition) wird die Erweiterung um statische Event-Attribute erläutert. Die Beschreibung umfasst Annahmen, mathematische Formulierungen, die resultierende Tabellenstruktur und ein kleines, veranschaulichendes Beispiel.

## Eingangsdaten und Notation
- Eingang: XES-Eventlog, importiert via PM4Py (`create_eventlog`).
- Ein Trace ist eine zeitlich geordnete Folge von Events; minimale Attribute:
  - `concept:name` (string): Eventtyp.
  - `time:timestamp` (datetime): Zeitstempel.
- Optionale statische Event-Attribute: `org:resource`, `org:role` (erweiterbar).

Notation:
- \(n_{traces}\): Anzahl Traces.
- \(K\): maximale Trace-Länge (maximale Zahl an Events pro Trace im Log).
- \(n_{events}\): Zahl eindeutiger Eventtypen (aus `concept:name`).

## Event-ID-basiertes Sequenz-Encoding (One-Hot)
Ziel ist eine feste Sequenzlänge je Trace und ein One-Hot-Vektor pro Eventposition über die globale Eventvokabularmenge.

1) Vokabularbildung: Alphabetisch sortierte Menge eindeutiger `concept:name`-Werte (Determinismus).
2) Chronologische Ordnung: Events je Trace anhand `time:timestamp` aufsteigend sortieren.
3) Tensorisierung: \(E \in \{0,1\}^{n_{traces} \times K \times n_{events}}\).
   - Für Trace \(i\) und Position \(j\) (0-basiert) wird genau ein Eintrag 1, alle anderen 0.
   - Traces mit Länge < \(K\) werden rechts mit Nullen aufgefüllt (Padding).
4) Zielvariable (Dauer): \(d_i = \max_t(\text{timestamp}) - \min_t(\text{timestamp})\) in Sekunden; z-Score-Normalisierung, falls Varianz > 0.

Ausgabe:
- 3D-Tensor `encoded_sequences` mit Form `[n_traces, K, n_events]`.
- Vektor `normalized_durations` der Länge `n_traces`.
- `trace_ids` (Bezeichner je Trace).

### Flattening in ein DataFrame
Für tabellarische Modelle (z. B. XGBoost) wird der Tensor in ein DataFrame abgeflacht:
- Spalten: für jede Position `pos ∈ {1,…,K}` und jeden Eventnamen `e`: `Event_{pos}_{e}` (0/1).
- Zielvariable: `Total_Duration_Normalized` (float, z-Score).
- Identifikator: `Trace_ID` als erste Spalte.

Diese Transformation wird von `convert_to_dataframe` implementiert; die Erzeugung des 3D-Tensors übernimmt `build_dataframe`.

## Erweiterung: Statische Event-Attribute
Statische, kategoriale Event-Attribute (z. B. `org:resource`, `org:role`, `concept:name`) werden pro Eventposition ein-hot kodiert und an die tabellarische Repräsentation angefügt.

1) Attributauswahl: Standardmäßig werden diejenigen Attribute aus dem Kandidatenset `['org:resource','org:role','concept:name']` gewählt, die in jedem Trace des Datensatzes mindestens einmal vorkommen (robuste Standardwahl). Optional kann eine explizite Attributliste übergeben werden (`attribute_names` in `build_static_features_dataframe` bzw. `load_encoded_with_static_attributes`).
2) Klassenbildung: Für jedes Attribut werden eindeutige Werte gesammelt und alphabetisch sortiert.
3) One-Hot je Position: Für Position `pos` und Attribut `a` entstehen Spalten `Event_{pos}_{a}_{klasse}`.
4) Dauer: Die statische Matrix enthält `Total_Duration` (unnormalisiert). Für die kombinierte Sicht wird diese Spalte verworfen und die z-normalisierte Zielvariable aus dem Event-ID-DataFrame verwendet.

Die statische Featurematrix wird von `build_static_features_dataframe` erzeugt.

### Kombination beider Sichten
Die Funktion `load_encoded_with_static_attributes(xes_path, attribute_names=None)` produziert einen DataFrame, der das Event-ID-Flattening mit den statischen One-Hot-Blöcken je Position zusammenführt:
- Erzeugt `encoded_sequences` und `convert_to_dataframe` (Event-IDs + `Total_Duration_Normalized`).
- Erzeugt statische One-Hot-Features je Position (ohne `Total_Duration`).
- Merged auf `Trace_ID` und füllt fehlende Werte mit 0.

Damit stehen zwei tabellarische Varianten zur Verfügung:
- Ohne statische Attribute: nur `Event_{pos}_{e}` + `Total_Duration_Normalized`.
- Mit statischen Attributen: zusätzlich `Event_{pos}_{attr}_{klasse}`.

## Beispiel (veranschaulichend)
Angenommen:
- Vokabular der Eventtypen: `['A','B']` (\(n_{events}=2\)).
- Maximale Sequenzlänge: \(K=3\).
- Ein Trace mit chronologischer Folge `['B','A']` und ein zweiter Trace nur `['A']`.
- Statisches Attribut `org:role ∈ {'Clerk','Manager'}`.

### 3D-Tensor (One-Hot, Schema)
Für Trace 1 (Index 0):
```
pos\event    A  B
1            0  1
2            1  0
3            0  0   # Padding
```
Für Trace 2 (Index 1):
```
pos\event    A  B
1            1  0
2            0  0   # Padding
3            0  0   # Padding
```

### Abgeflachter DataFrame (ohne statische Attribute)
Spalten (Auszug):
```
Trace_ID, Event_1_A, Event_1_B, Event_2_A, Event_2_B, Event_3_A, Event_3_B, Total_Duration_Normalized
case_001,        0,        1,        1,        0,        0,        0,                         0.37
case_002,        1,        0,        0,        0,        0,        0,                        -0.12
```

### Zusätzliche Spalten für statische Attribute (Schema)
Für `org:role ∈ {'Clerk','Manager'}` entstehen pro Position die Spalten:
```
Event_1_org:role_Clerk, Event_1_org:role_Manager,
Event_2_org:role_Clerk, Event_2_org:role_Manager,
Event_3_org:role_Clerk, Event_3_org:role_Manager,
```
Diese werden an den oben dargestellten DataFrame rechts angefügt; fehlende Informationen je Position (z. B. weil der Trace kürzer ist) bleiben 0.

## Reproduzierbarkeit und Randfälle
- Vokabular und Klassenlisten werden alphabetisch sortiert (deterministische Spaltenordnung).
- Dauerberechnung: Fehlen <2 Zeitstempel in einem Trace, wird die Dauer 0.0 gesetzt; z-Score wird nur angewandt, wenn die Varianz > 0 ist.
- Padding: Traces kürzer als \(K\) werden rechts mit Nullen aufgefüllt; dies spiegelt sich sowohl im 3D-Tensor als auch im abgeflachten DataFrame.

## API-Referenzen (Kernauswahl)
- `create_eventlog(xes_path)`: XES-Import via PM4Py.
- `build_dataframe(event_log)`: 3D One-Hot-Tensor + z-normalisierte Dauer.
- `convert_to_dataframe(encoded_sequences, normalized_durations, trace_ids, encoder)`: Flattening in tabellarische Form.
- `build_static_features_dataframe(event_log)`: statische One-Hot-Features je Eventposition.
- `load_encoded_with_static_attributes(xes_path)`: kombinierter DataFrame (Event-IDs + statische Attribute).

## Komplexität (vereinfacht)
- Zeit: \(\mathcal{O}(n_{traces} \cdot K \cdot n_{events})\) für das One-Hot-Encoding; statische Attribute fügen \(\sum_a |\mathcal{C}_a|\) pro Position hinzu.
- Speicher: \(\mathcal{O}(n_{traces} \cdot K \cdot n_{events})\) (ohne statische Attribute), plus \(\mathcal{O}(n_{traces} \cdot K \cdot \sum_a |\mathcal{C}_a|)\) mit statischen Attributen.

## Anwendungskontext
- Sequenzmodelle (CNN/ResNet/TCN/Transformer): nutzen typischerweise die 3D-Sequenzform `[N, K, M]` bzw. `[N, C, L]`.
- Tabulare Modelle (z. B. XGBoost): nutzen den abgeflachten DataFrame mit konsistenter Spaltenbenennung.
