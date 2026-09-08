#include <phy_engine/phy_lab_wrapper/physicslab.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

// Read-only electrical views. The schematic renderer uses conventional
// symbols and orthogonal node wiring; uncommon multi-pin devices use IEC
// functional blocks with exact pin labels. The adjacent JSON keeps full
// identifiers/properties (never OCR-derived).
namespace pl = phy_engine::phy_lab_wrapper;
using json = pl::json;

std::string escaped(std::string const& s)
{
    std::string out;
    for (char c : s)
        switch (c)
        {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: if (static_cast<unsigned char>(c) >= 32 || c == '\n' || c == '\t') out += c;
        }
    return out;
}

json read_json(char const* path)
{
    if (std::filesystem::file_size(path) > 32 * 1024 * 1024) throw std::runtime_error("input exceeds 32 MiB");
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read input");
    return json::parse(in, nullptr, true, true);
}

pl::position required_xyz(std::string const& value, std::string const& label)
{
    try
    {
        auto parsed = pl::detail::parse_xyz(value);
        if (!std::isfinite(parsed.x) || !std::isfinite(parsed.y) || !std::isfinite(parsed.z))
            throw std::runtime_error("coordinates must be finite");
        return parsed;
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error("invalid xyz coordinates: " + label + ": " + error.what());
    }
}

std::size_t known_pins(std::string const& type)
{
    static std::map<std::string, std::size_t> const pins{
        {"Ground Component", 1}, {"Logic Input", 1}, {"Logic Output", 1},
        {"Resistor", 2}, {"Basic Capacitor", 2}, {"Basic Inductor", 2}, {"Battery Source", 2},
        {"Current Source", 2}, {"Sinewave Source", 2}, {"Basic Diode", 2}, {"Light-Emitting Diode", 2},
        {"Simple Switch", 2}, {"Push Switch", 2}, {"Air Switch", 2}, {"Yes Gate", 2}, {"No Gate", 2},
        {"And Gate", 3}, {"Or Gate", 3}, {"Xor Gate", 3}, {"Xnor Gate", 3}, {"Nand Gate", 3}, {"Nor Gate", 3},
        {"Imp Gate", 3}, {"Nimp Gate", 3}, {"Transistor", 3}, {"N-MOSFET", 3}, {"P-MOSFET", 3},
        {"Half Adder", 4}, {"Full Adder", 5}, {"Half Subtractor", 4}, {"Full Subtractor", 5},
        {"D Flipflop", 4}, {"T Flipflop", 4}, {"Real-T Flipflop", 4}, {"JK Flipflop", 5},
        {"Counter", 6}, {"Random Generator", 6}, {"8bit Input", 8}, {"8bit Display", 8},
        {"Transformer", 4}, {"Mutual Inductor", 4}, {"Rectifier", 4}, {"Multiplier", 8},
        // Pin-schema metadata only; numerical support is declared separately.
        {"555 Timer", 8}, {"Accelerometer", 3}, {"Analog Joystick", 6},
        {"Attitude Sensor", 3}, {"Buzzer", 2}, {"Color Light-Emitting Diode", 4},
        {"Comparator", 3}, {"DPDT Switch", 6}, {"Dual Light-Emitting Diode", 2},
        {"Eight Bit Display", 8}, {"Eight Bit Input", 8}, {"Electric Bell", 2},
        {"Electric Fan", 2}, {"Electricity Meter", 4}, {"Fuse Component", 2},
        {"Galvanometer", 3}, {"Gravity Sensor", 3}, {"Gyroscope", 3},
        {"Incandescent Lamp", 2}, {"Linear Accelerometer", 3}, {"Magnetic Field Sensor", 3},
        {"Microammeter", 3}, {"Multimeter", 2}, {"Musical Box", 2},
        {"Operational Amplifier", 3}, {"Photodiode", 2}, {"Photoresistor", 2},
        {"Proximity Sensor", 1}, {"Pulse Source", 2}, {"Relay Component", 5},
        {"Resistance Box", 2}, {"Resistance Law", 8}, {"SPDT Switch", 3},
        {"Sawtooth Source", 2}, {"Schmitt Trigger", 2}, {"Simple Ammeter", 3},
        {"Simple Instrument", 2}, {"Simple Voltmeter", 3}, {"Slide Rheostat", 4},
        {"Solenoid", 4}, {"Spark Gap", 2}, {"Square Source", 2}, {"Student Source", 4},
        {"Tapped Transformer", 5}, {"Tesla Coil", 2}, {"Triangle Source", 2}};
    auto it = pins.find(type);
    return it == pins.end() ? 0 : it->second;
}

json inspect(pl::experiment const& ex)
{
    if (ex.type() != pl::experiment_type::circuit) throw std::runtime_error("only electrical experiments are supported");
    if (ex.elements().size() > 20000 || ex.wires().size() > 80000) throw std::runtime_error("circuit exceeds inspect limits");
    json result{{"schema", "aurex.circuit-view.v1"}, {"components", json::array()}, {"nodes", json::array()},
                {"wires", json::array()}, {"warnings", json::array()},
                {"statistics_source", "saved values, not fresh simulation measurements"}};
    std::map<std::string, std::size_t> index;
    std::map<std::string, std::size_t> declared_pins;
    std::map<std::pair<std::string, int>, std::size_t> terminals;
    std::vector<std::size_t> parent;
    auto terminal = [&](std::string const& id, int pin) {
        if (!index.contains(id) || pin < 0 || pin > 255) throw std::runtime_error("invalid wire endpoint: " + id);
        if (declared_pins.contains(id) && declared_pins.at(id) && static_cast<std::size_t>(pin) >= declared_pins.at(id))
            throw std::runtime_error("wire endpoint exceeds known pin schema: " + id);
        auto key = std::make_pair(id, pin);
        if (!terminals.contains(key)) { terminals[key] = parent.size(); parent.push_back(parent.size()); }
        return terminals.at(key);
    };
    auto root = [&](std::size_t n) {
        while (parent[n] != n) { parent[n] = parent[parent[n]]; n = parent[n]; }
        return n;
    };
    for (auto const& e : ex.elements())
    {
        auto d = e.data();
        auto id = e.identifier();
        index[id] = result["components"].size();
        auto model = d.value("ModelID", "");
        std::size_t count = known_pins(model);
        if (d.contains("Aurex") && d["Aurex"].is_object()) count = d["Aurex"].value("pin_count", count);
        if (count > 256) throw std::runtime_error("too many component pins");
        declared_pins[id] = count;
        auto label = d.contains("Label") && d["Label"].is_string() ? d["Label"].get<std::string>() : "";
        auto pos = e.element_position();
        bool has_position = d.contains("Position") && d["Position"].is_string() && !d["Position"].get<std::string>().empty();
        if (!has_position) pos = {static_cast<double>(index[id] % 4) * .18, static_cast<double>(index[id] / 4) * .14, 0};
        if (has_position) required_xyz(d["Position"].get<std::string>(), id + " Position");
        auto rot = d.contains("Rotation") && d["Rotation"].is_string() ? required_xyz(d["Rotation"].get<std::string>(), id + " Rotation") : pl::position{};
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) || !std::isfinite(rot.x) || !std::isfinite(rot.y) || !std::isfinite(rot.z)) throw std::runtime_error("nonfinite position/rotation: " + id);
        result["components"].push_back(json{{"id", id}, {"ref", "C" + std::to_string(index[id] + 1)},
            {"label", label}, {"type", model}, {"properties", d.value("Properties", json::object())},
            {"statistics", d.value("Statistics", json::object())}, {"pins", json::array()},
            {"is_broken", d.value("IsBroken", json(false))}, {"raw_element", d},
            {"native", d.value("Aurex", json::object())}, {"pin_count_known", count != 0},
            {"position", {pos.x, pos.y, pos.z}}, {"rotation", {rot.x, rot.y, rot.z}},
            {"position_source", has_position ? d.value("Aurex", json::object()).value("position_source", "saved") : "generated"}});
        for (std::size_t p = 0; p < count; ++p) terminal(id, static_cast<int>(p));
        if (!count) result["warnings"].push_back("Unknown pin schema: " + id + " (" + model + "); only wired pins shown");
    }
    for (auto const& w : ex.wires())
    {
        auto a = terminal(w.source.element_identifier, w.source.pin);
        auto b = terminal(w.target.element_identifier, w.target.pin);
        parent[root(a)] = root(b);
        result["wires"].push_back(w.to_json());
    }
    std::map<std::size_t, std::string> net_names;
    for (auto const& [key, n] : terminals)
    {
        auto& c = result["components"][index.at(key.first)];
        if (c["type"] == "Ground Component") net_names[root(n)] = "gnd";
    }
    std::map<std::string, json> nets;
    for (auto const& [key, n] : terminals)
    {
        auto r = root(n);
        if (!net_names.contains(r)) net_names[r] = "N" + std::to_string(net_names.size() + 1);
        auto name = net_names[r];
        auto& c = result["components"][index.at(key.first)];
        c["pins"].push_back({{"pin", key.second}, {"node", name}});
        if (!nets.contains(name)) nets[name] = json::array();
        nets[name].push_back({{"component", key.first}, {"ref", c["ref"]}, {"pin", key.second}});
    }
    for (auto const& [name, ends] : nets) result["nodes"].push_back({{"id", name}, {"connections", ends}});
    return result;
}

// A solver snapshot is not a PhysicsLab document. Geometry is supplied by the
// native spec sidecar; topology comes directly from the sampled solver spec.
std::string component_label(json const& component)
{
    if (!component.contains("label") || component["label"].is_null()) return {};
    if (!component["label"].is_string()) throw std::runtime_error("component label must be a string or null");
    return component["label"].get<std::string>();
}

json inspect_state(json const& state)
{
    if (state.value("schema", "") != "aurex.pe-state.v1") throw std::runtime_error("state mode requires aurex.pe-state.v1");
    auto const& spec = state.at("spec");
    auto const& components = spec.at("components");
    // Rendering an immutable snapshot does not execute the solver. Its scene
    // budget must accommodate supported digital jobs; analog/mixed simulation
    // remains limited independently in both the tool and isolated worker.
    if (!components.is_array() || components.empty() || components.size() > 16384) throw std::runtime_error("native state render must contain 1..16384 components");
    std::map<std::string, json> scene, measurements;
    for (auto const& c : state.at("scene").at("components")) scene.emplace(c.at("id").get<std::string>(), c);
    for (auto const& c : state.at("measurements").at("components")) measurements.emplace(c.at("id").get<std::string>(), c);
    json out{{"schema", "aurex.circuit-view.v1"}, {"components", json::array()}, {"nodes", json::array()}, {"wires", json::array()},
        {"warnings", json::array()}, {"statistics_source", "immutable snapshot sampled from a native Phy-Engine handle; not saved PL statistics"},
        {"state_origin", state.value("origin", json::object())}, {"state_schema", "aurex.pe-state.v1"}};
    std::map<std::string, json> nodes;
    std::set<std::string> ids;
    for (auto const& c : components)
    {
        auto id = c.at("id").get<std::string>();
        if (id.empty() || !ids.insert(id).second || !scene.contains(id) || !measurements.contains(id)) throw std::runtime_error("snapshot IDs must be unique and have matching scene/measurement entries");
        auto const& s = scene.at(id);
        if (s.at("nodes") != c.at("nodes") || s.at("position") != c.at("position") || s.at("rotation") != c.at("rotation")) throw std::runtime_error("snapshot scene must preserve native spec nodes/position/rotation");
        auto label = component_label(c);
        if (s.contains("label") && component_label(s) != label) throw std::runtime_error("snapshot scene label must match native spec label");
        if (!c.at("nodes").is_array() || c.at("nodes").empty() || c.at("nodes").size() > 256) throw std::runtime_error("invalid native snapshot pins");
        for (auto name : {"position", "rotation"})
        {
            if (!c.at(name).is_array() || c.at(name).size() != 3) throw std::runtime_error("snapshot position/rotation must be xyz triples");
            for (auto const& v : c.at(name)) if (!v.is_number() || !std::isfinite(v.get<double>())) throw std::runtime_error("snapshot contains nonfinite geometry");
        }
        auto native = s.value("native", json::object());
        native["type"] = c.at("type");
        native["measurements"] = measurements.at(id);
        auto ref = "C" + std::to_string(out["components"].size() + 1);
        json pins = json::array();
        int pin = 0;
        for (auto const& n : c.at("nodes"))
        {
            auto node = n.get<std::string>();
            if (node.empty()) throw std::runtime_error("snapshot node names must not be empty");
            if (!nodes.contains(node)) nodes[node] = json::array();
            pins.push_back({{"node", node}, {"pin", pin}});
            nodes[node].push_back({{"component", id}, {"ref", ref}, {"pin", pin++}});
        }
        out["components"].push_back({{"id", id}, {"ref", ref}, {"label", label}, {"type", s.at("model_id")}, {"properties", s.value("properties", json::object())},
            {"statistics", json::object()}, {"pins", pins}, {"native", native}, {"pin_count_known", true}, {"position", c.at("position")}, {"rotation", c.at("rotation")},
            {"position_source", c.value("position_source", "native-spec-sidecar")}});
    }
    for (auto const& [node, pins] : nodes)
    {
        out["nodes"].push_back({{"id", node}, {"connections", pins}});
        for (std::size_t i = 1; i < pins.size(); ++i) out["wires"].push_back({{"Source", pins[0]["component"]}, {"SourcePin", pins[0]["pin"]},
            {"Target", pins[i]["component"]}, {"TargetPin", pins[i]["pin"]}, {"ColorName", "native-node"}});
    }
    return out;
}

pl::experiment from_spec(json const& spec)
{
    auto ex = pl::experiment::create(pl::experiment_type::circuit);
    ex.entitle(spec.value("title", "Aurex circuit"));
    auto const& components = spec.at("components");
    if (!components.is_array() || components.empty() || components.size() > 16384) throw std::runtime_error("render/export scene must contain 1..16384 entries");
    std::map<std::string, std::vector<std::pair<std::string, int>>> nodes;
    std::size_t i = 0;
    for (auto const& c : components)
    {
        pl::position position{static_cast<double>(i % 4) * .18, static_cast<double>(i / 4) * .14, 0};
        if (c.contains("position")) position = {c["position"][0], c["position"][1], c["position"][2]};
        auto e = pl::element::circuit(c.at("model_id"), position);
        e.write_native_position(position);
        if (c.contains("rotation")) e.set_rotation({c["rotation"][0], c["rotation"][1], c["rotation"][2]});
        auto id = c.at("id").get<std::string>();
        e.data()["Identifier"] = id;
        // Identifier owns connectivity. Label is optional user-facing metadata
        // and must never be silently invented or substituted for Identifier.
        static_cast<void>(component_label(c));
        e.data()["Label"] = c.value("label", json(nullptr));
        e.data()["Properties"].update(c.value("properties", json::object()));
        e.data()["Aurex"] = c.value("native", json::object());
        e.data()["Aurex"]["pin_count"] = c.at("nodes").size();
        e.data()["Aurex"]["position_source"] = c.value("native", json::object()).value("position_source", c.contains("position") ? "provided" : "generated");
        static_cast<void>(ex.add_element(std::move(e)));
        int p = 0;
        for (auto const& n : c.at("nodes")) nodes[n.get<std::string>()].push_back({id, p++});
        ++i;
    }
    for (auto const& [name, ends] : nodes)
    {
        if (name == "gnd")
        {
            auto ground = ex.add_circuit_element("Ground Component", {0, -.1, 0});
            ex.get_element(ground).write_native_position({0, -.1, 0});
            ex.get_element(ground).data()["Aurex"] = {{"position_source", "generated"}};
            for (auto const& [id, p] : ends) ex.connect(ground, 0, id, p);
        }
        else for (std::size_t k = 1; k < ends.size(); ++k) ex.connect(ends[0].first, ends[0].second, ends[k].first, ends[k].second);
    }
    return ex;
}

void render_topology(json const& data, std::ostream& out, std::size_t offset, std::size_t limit)
{
    auto const& components = data.at("components");
    std::size_t end = std::min(components.size(), offset + limit);
    if (offset >= components.size()) throw std::runtime_error("page offset exceeds component count");
    std::map<std::string, std::size_t> nets;
    std::vector<int> tops;
    int height = 116;
    for (std::size_t i = offset; i < end; ++i)
    {
        tops.push_back(height);
        height += std::max(137, 56 + static_cast<int>(components[i]["pins"].size()) * 23);
        for (auto const& p : components[i]["pins"]) nets.try_emplace(p["node"].get<std::string>(), nets.size());
    }
    if (nets.size() > 64) throw std::runtime_error("view contains more than 64 nets; decrease page limit");
    int width = std::max(900, 560 + static_cast<int>(nets.size()) * 52);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height + 35 << "\" viewBox=\"0 0 " << width << ' ' << height + 35 << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/><g font-family=\"DejaVu Sans, sans-serif\" fill=\"#0f172a\">";
    auto text = [&](int x, int y, std::string const& s, int size = 15) {
        out << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << size << "\">" << escaped(s) << "</text>\n";
    };
    text(24, 30, "Aurex | electrical netlist", 23);
    text(24, 56, "Ports are zero-based. Filled dots = junctions; crossings without dots are not connected.", 14);
    text(24, 78, (data.value("focused", false) ? "Focused components " : "Components ") + std::to_string(offset + 1) + "-" + std::to_string(end) + "/" + std::to_string(data.value("total_components", components.size())) + "; full IDs and all values in netlist JSON.", 13);
    char const* colors[]{"#2563eb", "#b45309", "#047857", "#9333ea", "#be123c", "#0e7490", "#4d7c0f", "#9a3412"};
    for (auto const& [name, n] : nets)
    {
        int x = 520 + static_cast<int>(n) * 52;
        out << "<path d=\"M " << x << " 105 V " << height << "\" stroke=\"" << colors[n % 8] << "\" stroke-width=\"2\" opacity=\".55\"/>";
        text(x - 12, 100, name, 13);
    }
    for (std::size_t i = offset; i < end; ++i)
    {
        auto const& c = components[i];
        int y = tops[i - offset];
        int h = std::max(119, 38 + static_cast<int>(c["pins"].size()) * 23);
        out << "<rect x=\"24\" y=\"" << y << "\" width=\"350\" height=\"" << h << "\" rx=\"8\" fill=\"white\" stroke=\"#94a3b8\"/>";
        auto label = c.value("label", "");
        text(38, y + 24, c["ref"].get<std::string>() + "  " + (label.empty() ? c["type"].get<std::string>() : label), 17);
        text(38, y + 46, c["type"].get<std::string>(), 14);
        std::string values;
        auto const& props = c["properties"];
        std::map<std::string, std::string> names{{"电阻", "R (ohm)"}, {"电容", "C (F)"}, {"电感", "L (H)"}, {"电压", "V (V)"}, {"电流", "I (A)"}, {"开关", "state"},
            {"内阻", "R_internal (ohm)"}, {"理想模式", "ideal"}, {"十进制", "value"}, {"状态", "state"},
            {"放大系数", "beta"}, {"最大功率", "max power (W)"}};
        for (auto const& [k, v] : props.items())
        {
            if (k == "锁定" || k == "高电平" || k == "低电平" || k == "最大电流") continue;
            auto key = names.contains(k) ? names[k] : k;
            if (!values.empty()) values += ", ";
            values += key + "=" + v.dump();
        }
        if (values.size() > 50)
        {
            auto cut = std::size_t{47};
            while (cut && (static_cast<unsigned char>(values[cut]) & 0xc0) == 0x80) --cut;
            values = values.substr(0, cut) + "...";
        }
        text(38, y + 70, values, 13);
        if (c["native"].contains("measurements"))
        {
            auto const& m = c["native"]["measurements"];
            std::ostringstream measured;
            measured.precision(4);
            if (c["native"].value("type", "").starts_with("digital_"))
            {
                measured << "Measured logic: ";
                for (auto const& v : m.at("digital")) measured << "01XZ"[v.get<unsigned>() % 4] << ' ';
            }
            else
            {
                int pin = 0;
                for (auto const& v : m.at("voltage"))
                {
                    if (pin && pin % 2 == 0)
                    {
                        text(38, y + 94 + (pin / 2 - 1) * 18, measured.str(), 12);
                        measured.str("");
                        measured.clear();
                    }
                    else if (pin) measured << "; ";
                    measured << "V(p" << pin << ")=" << v.get<double>();
                    double imag = m.at("voltage_imag")[pin].get<double>();
                    if (imag != 0) measured << (imag < 0 ? "-j" : "+j") << std::abs(imag);
                    measured << " V";
                    ++pin;
                }
                int line = pin ? (pin - 1) / 2 : 0;
                text(38, y + 94 + line * 18, measured.str(), 12);
                measured.str("");
                measured.clear();
                if (m.contains("derived_current_0_to_1"))
                {
                    auto const& cur = m["derived_current_0_to_1"];
                    measured << "I(p0 to p1)=" << cur["real"].get<double>();
                    double imag = cur["imag"].get<double>();
                    if (imag != 0) measured << (imag < 0 ? "-j" : "+j") << std::abs(imag);
                    measured << " A";
                    text(38, y + 112, measured.str(), 12);
                }
            }
            if (c["native"].value("type", "").starts_with("digital_")) text(38, y + 94, measured.str(), 12);
        }
        int pidx = 0;
        for (auto const& p : c["pins"])
        {
            auto name = p["node"].get<std::string>();
            auto n = nets.at(name);
            int py = y + 20 + pidx++ * 23, nx = 520 + static_cast<int>(n) * 52;
            out << "<path d=\"M 374 " << py << " H " << nx << "\" stroke=\"" << colors[n % 8] << "\" stroke-width=\"2\"/>";
            out << "<circle cx=\"" << nx << "\" cy=\"" << py << "\" r=\"4\" fill=\"" << colors[n % 8] << "\"/>";
            text(381, py - 4, "p" + std::to_string(p["pin"].get<int>()) + " / " + name, 12);
        }
    }
    out << "</g></svg>\n";
}

struct xy { double x, y; };

pl::position add(pl::position a, pl::position b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
pl::position mul(pl::position p, double k) { return {p.x * k, p.y * k, p.z * k}; }
double dot(pl::position a, pl::position b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
pl::position cross(pl::position a, pl::position b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
pl::position unit(pl::position p)
{
    double length = std::sqrt(dot(p, p));
    if (!std::isfinite(length) || length < 1e-10) throw std::runtime_error("camera position/target must define a nonzero direction");
    return mul(p, 1 / length);
}
json xyz_json(pl::position p) { return {p.x, p.y, p.z}; }
pl::position checked_xyz(json const& value, std::string const& name)
{
    if (!value.is_array() || value.size() != 3) throw std::runtime_error(name + " must be a native xyz array");
    for (auto const& v : value) if (!v.is_number() || !std::isfinite(v.get<double>()) || std::abs(v.get<double>()) > 1e8) throw std::runtime_error(name + " contains an invalid coordinate");
    return {value[0], value[1], value[2]};
}

json camera_save(json const& input, bool native_state, json& warnings)
{
    json raw = native_state || input.contains("camera_save") ? input.value("camera_save", json()) : input.value("Experiment", input).value("CameraSave", json());
    try
    {
        if (raw.is_string())
        {
            if (raw.get<std::string>().empty()) return json::object();
            raw = json::parse(raw.get<std::string>());
        }
        if (raw.is_null()) return json::object();
        if (!raw.is_object()) throw std::runtime_error("CameraSave must be an object or JSON-encoded object");
        return raw;
    }
    catch (std::exception const& e)
    {
        warnings.push_back(std::string("Cannot decode saved camera; using fallback: ") + e.what());
        return json::object();
    }
}

struct view_camera
{
    pl::position right{.707106781187, -.707106781187, 0};
    pl::position up{-.408248290464, -.408248290464, .816496580928};
    pl::position forward{-.577350269190, -.577350269190, -.577350269190};
    pl::position position{}, target{}, rotation{};
    double distance{2}, fov{60}, zoom{1}, ortho_height{1}, near_plane{1e-4};
    bool perspective{false}, fit{true}, rotation_available{false};
    std::string source{"fallback-isometric"};
    json saved = json::object(), warnings = json::array(), assumptions = json::array();

    void euler_basis(pl::position native)
    {
        // PhysicsLab serializes native x,z,y. Its SDK does not document the
        // client Euler composition; ZXY is an explicit Unity-style renderer
        // approximation, never reported as exact client-camera replication.
        rotation = native;
        rotation_available = true;
        double x = native.x * std::acos(-1.0) / 180;
        double y = native.z * std::acos(-1.0) / 180;
        double z = native.y * std::acos(-1.0) / 180;
        auto rotate = [&](pl::position p) {
            p = {std::cos(z)*p.x-std::sin(z)*p.y, std::sin(z)*p.x+std::cos(z)*p.y, p.z};
            p = {p.x, std::cos(x)*p.y-std::sin(x)*p.z, std::sin(x)*p.y+std::cos(x)*p.z};
            p = {std::cos(y)*p.x+std::sin(y)*p.z, p.y, -std::sin(y)*p.x+std::cos(y)*p.z};
            return pl::position{p.x, p.z, p.y};
        };
        right = rotate({1,0,0}); up = rotate({0,1,0}); forward = rotate({0,0,1});
        assumptions.push_back("TargetRotation uses renderer Unity-style ZXY Euler approximation; PhysicsLab client composition is not documented");
    }
    void look_at()
    {
        rotation_available = false;
        assumptions = json::array();
        forward = unit(add(target, mul(position, -1)));
        pl::position reference_up{0,0,1};
        if (std::abs(dot(forward, reference_up)) > .999) reference_up = {0,1,0};
        right = unit(cross(forward, reference_up));
        up = unit(cross(right, forward));
        distance = std::sqrt(dot(add(position, mul(target, -1)), add(position, mul(target, -1))));
    }
    double depth(pl::position p) const { return dot(add(p, mul(position, -1)), forward); }
    json metadata() const
    {
        return {{"source", source}, {"coordinate_order", "native xyz; CameraSave strings serialize x,z,y"}, {"distance_unit", "native coordinate units"},
            {"saved_raw", saved}, {"position", xyz_json(position)}, {"target", xyz_json(target)}, {"rotation", rotation_available ? xyz_json(rotation) : json()},
            {"right", xyz_json(right)}, {"up", xyz_json(up)}, {"forward", xyz_json(forward)}, {"distance", distance}, {"fov_y_deg", fov},
            {"projection", perspective ? "perspective" : "orthographic"}, {"orthographic_height", ortho_height}, {"zoom", zoom},
            {"fit", fit}, {"near_plane", near_plane}, {"assumptions", assumptions}, {"warnings", warnings}};
    }
};

view_camera make_camera(json const& saved, json const& options, std::vector<pl::position> const& points, bool top, int height, double frame_width = 530, double frame_height = 0)
{
    if (!options.is_object()) throw std::runtime_error("camera options must be an object");
    std::set<std::string> const keys{"mode", "position", "target", "rotation", "distance", "yaw_deg", "pitch_deg", "zoom", "fov_y_deg", "projection", "orthographic_height", "fit", "overview"};
    for (auto const& [key, ignored] : options.items()) if (!keys.contains(key)) throw std::runtime_error("Unknown camera option: " + key);
    auto number = [&](char const* name, double fallback, double lo, double hi) {
        if (!options.contains(name)) return fallback;
        auto const& value = options.at(name);
        if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() < lo || value.get<double>() > hi) throw std::runtime_error(std::string("Invalid camera ") + name);
        return value.get<double>();
    };
    view_camera c;
    c.saved = saved;
    std::string mode = options.value("mode", "saved");
    if (mode != "saved" && mode != "auto" && mode != "custom") throw std::runtime_error("camera mode must be saved, auto or custom");
    bool saved_valid = false;
    if (!saved.empty() && mode != "auto" && !top)
    {
        try
        {
            auto center = required_xyz(saved.at("VisionCenter").get<std::string>(), "saved VisionCenter");
            auto rotation = required_xyz(saved.at("TargetRotation").get<std::string>(), "saved TargetRotation");
            c.target = checked_xyz(xyz_json(center), "saved VisionCenter");
            checked_xyz(xyz_json(rotation), "saved TargetRotation");
            if (!saved.at("Distance").is_number()) throw std::runtime_error("saved Distance is not numeric");
            c.distance = saved.at("Distance").get<double>();
            if (!std::isfinite(c.distance) || c.distance <= 1e-4 || c.distance > 1e8) throw std::runtime_error("saved Distance is invalid");
            c.euler_basis(rotation);
            c.source = "saved"; c.perspective = true; c.fit = false; saved_valid = true;
            c.assumptions.push_back("CameraSave has no documented FOV/projection mapping; perspective and 60-degree vertical FOV are renderer defaults, not recovered client settings");
            c.assumptions.push_back("Saved camera is world-space but its origin relative to component native coordinates is undocumented; no hidden tabletop offset is applied");
            if (saved.contains("Mode")) c.assumptions.push_back("CameraSave Mode identifies experiment kind, not orthographic/perspective projection");
        }
        catch (std::exception const& e) { c.warnings.push_back(std::string("Saved camera incomplete/invalid; using fitted isometric fallback: ") + e.what()); }
    }
    if (top && !options.contains("mode") && !options.contains("position") && !options.contains("rotation") && !options.contains("yaw_deg") && !options.contains("pitch_deg"))
    {
        c.right = {1,0,0}; c.up = {0,1,0}; c.forward = {0,0,-1};
        c.source = "explicit-top"; c.perspective = false; c.fit = true;
    }
    if (mode == "auto") { c.source = "auto-fit"; c.fit = true; }
    if (mode == "custom" || options.contains("position") || options.contains("target") || options.contains("rotation") || options.contains("yaw_deg") || options.contains("pitch_deg"))
    {
        c.source = "custom";
        c.fit = mode == "auto";
        if (!saved_valid && !options.contains("projection")) c.perspective = true;
    }
    if (options.contains("fit"))
    {
        if (!options["fit"].is_boolean()) throw std::runtime_error("camera fit must be boolean");
        c.fit = options["fit"].get<bool>();
    }
    c.fov = number("fov_y_deg", c.fov, 5, 150);
    c.zoom = number("zoom", 1, .05, 20);
    c.distance = number("distance", c.distance, 1e-4, 1e8);
    if (options.contains("projection"))
    {
        auto projection = options.at("projection").get<std::string>();
        if (projection != "perspective" && projection != "orthographic") throw std::runtime_error("camera projection must be perspective or orthographic");
        c.perspective = projection == "perspective";
    }
    if (options.contains("target")) c.target = checked_xyz(options["target"], "camera target");
    bool has_euler = options.contains("rotation") || options.contains("yaw_deg") || options.contains("pitch_deg");
    if (options.contains("position") && (has_euler || options.contains("distance"))) throw std::runtime_error("camera position+target cannot also specify rotation/yaw/pitch/distance");
    if (has_euler)
    {
        auto rotation = options.contains("rotation") ? checked_xyz(options["rotation"], "camera rotation") : (saved_valid ? c.rotation : pl::position{50,0,0});
        rotation.x = number("pitch_deg", rotation.x, -36000, 36000);
        rotation.z = number("yaw_deg", rotation.z, -36000, 36000);
        c.euler_basis(rotation);
    }
    if (options.contains("position"))
    {
        if (!options.contains("target") && !saved_valid) throw std::runtime_error("custom camera position requires an explicit target when no saved target exists");
        c.position = checked_xyz(options["position"], "camera position");
        c.look_at();
        c.assumptions.push_back("Explicit position/target uses a deterministic look-at basis with native +Z up, avoiding saved Euler ambiguity");
    }
    else c.position = add(c.target, mul(c.forward, -c.distance));
    double h = frame_height > 0 ? frame_height : height - 290, w = frame_width;
    double tan_y = std::tan(c.fov * std::acos(-1.0) / 360), tan_x = tan_y * w / h;
    c.ortho_height = number("orthographic_height", 2 * c.distance * tan_y, 1e-5, 1e8);
    if (c.fit)
    {
        pl::position lo{1e99,1e99,1e99}, hi{-1e99,-1e99,-1e99};
        for (auto p : points) { lo = {std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)}; hi = {std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)}; }
        c.target = mul(add(lo,hi), .5);
        double distance = .2, max_x = .2, max_y = .12;
        for (auto p : points)
        {
            auto q = add(p, mul(c.target, -1));
            max_x = std::max(max_x, std::abs(dot(q,c.right)));
            max_y = std::max(max_y, std::abs(dot(q,c.up)));
            distance = std::max(distance, std::max(std::abs(dot(q,c.right))/tan_x, std::abs(dot(q,c.up))/tan_y) - dot(q,c.forward));
        }
        c.distance = distance * 1.08;
        c.ortho_height = std::max(max_y * 2, max_x * 2 * h / w) * 1.08;
        c.position = add(c.target, mul(c.forward,-c.distance));
        if (saved_valid) c.source = "saved-orientation-fitted";
        c.warnings.push_back("fit adjusted camera target/distance/framing to selected geometry; original saved camera and every component transform remain unchanged");
    }
    return c;
}

// Geometry is schematic; the component center and Euler rotation are the
// original PL native xyz, decoded by physicslab.h (serialized x,z,y).
pl::position local_to_world(json const& c, pl::position p)
{
    auto const& rot = c.at("rotation");
    double ax = rot[0].get<double>() * std::acos(-1.0) / 180;
    double ay = rot[1].get<double>() * std::acos(-1.0) / 180;
    double az = rot[2].get<double>() * std::acos(-1.0) / 180;
    p = {p.x, std::cos(ax) * p.y - std::sin(ax) * p.z, std::sin(ax) * p.y + std::cos(ax) * p.z};
    p = {std::cos(ay) * p.x + std::sin(ay) * p.z, p.y, -std::sin(ay) * p.x + std::cos(ay) * p.z};
    p = {std::cos(az) * p.x - std::sin(az) * p.y, std::sin(az) * p.x + std::cos(az) * p.y, p.z};
    return {p.x + c["position"][0].get<double>(), p.y + c["position"][1].get<double>(), p.z + c["position"][2].get<double>()};
}

pl::position port_position(json const& c, int pin)
{
    auto count = c.at("pins").size();
    if (count <= 1) return local_to_world(c, {-.049, 0, .018});
    if (count == 2) return local_to_world(c, {pin == 0 ? -.049 : .049, 0, .018});
    auto left_count = (count + 1) / 2;
    bool left = static_cast<std::size_t>(pin) < left_count;
    auto row = left ? pin : pin - static_cast<int>(left_count);
    auto rows = left ? left_count : count - left_count;
    double y = rows < 2 ? 0 : -.021 + .042 * static_cast<double>(row) / static_cast<double>(rows - 1);
    return local_to_world(c, {left ? -.049 : .049, y, .018});
}

std::string compact_properties(json const& c)
{
    std::map<std::string, std::string> names{{"电阻", "R"}, {"电容", "C"}, {"电感", "L"}, {"电压", "V"},
        {"电流", "I"}, {"开关", "stored state"}, {"内阻", "R_internal"}, {"十进制", "value"}, {"状态", "stored state"},
        {"放大系数", "beta"}, {"增益系数", "gain"}, {"额定电阻", "R_rated"},
        {"电阻1", "R1"}, {"电阻2", "R2"}, {"滑块位置", "wiper"},
        {"频率", "frequency"}, {"幅度", "amplitude"}, {"初相位", "phase"},
        {"占空比", "duty"}, {"最大功率", "max power"}};
    std::map<std::string, std::string> units{{"电阻", " ohm"}, {"电容", " F"}, {"电感", " H"}, {"电压", " V"}, {"电流", " A"}, {"内阻", " ohm"}};
    std::string result;
    int shown = 0;
    for (auto const& [key, value] : c["properties"].items())
    {
        if (key == "锁定" || key == "内阻" || key == "理想模式" || key == "高电平" || key == "低电平" ||
            key == "最大电流" || key == "最大功率" || !names.contains(key)) continue;
        if (shown++ == 2) { result += " ..."; break; }
        if (!result.empty()) result += "; ";
        result += names[key] + "=" + value.dump() + units[key];
    }
    return result;
}

// Trim display labels without splitting a UTF-8 code point. Several
// PhysicsLab property names and labels are Chinese; byte-wise resize() can
// leave an invalid continuation byte in the SVG and make the whole image
// unparsable.
std::string utf8_prefix(std::string const& value, std::size_t max_codepoints)
{
    std::size_t offset = 0, count = 0;
    while (offset < value.size() && count < max_codepoints)
    {
        auto byte = static_cast<unsigned char>(value[offset]);
        std::size_t width = byte < 0x80 ? 1 : byte < 0xe0 ? 2 : byte < 0xf0 ? 3 : 4;
        if (offset + width > value.size()) break;
        offset += width;
        ++count;
    }
    return value.substr(0, offset);
}

std::string utf8_ellipsize(std::string const& value, std::size_t max_codepoints)
{
    auto prefix = utf8_prefix(value, max_codepoints);
    return prefix.size() == value.size() ? value : prefix + "...";
}

// A low-detail locator, projected once from the complete original scene. It is
// not a second render/solve, and it never discards distant components to make
// the selection look larger. Wires connect centers here, not physical pins.
json render_locator(json const& global, std::set<std::string> const& visible, std::ostream& out, double x, double y, double width, double height)
{
    auto const& components = global.at("components");
    view_camera basis;
    double xmin=1e99,xmax=-1e99,ymin=1e99,ymax=-1e99;
    std::map<std::string,xy> projected;
    for(auto const& c:components)
    {
        auto center=local_to_world(c,{});
        xy p{dot(center,basis.right),-dot(center,basis.up)};
        projected.emplace(c["id"].get<std::string>(),p);
        xmin=std::min(xmin,p.x);xmax=std::max(xmax,p.x);ymin=std::min(ymin,p.y);ymax=std::max(ymax,p.y);
    }
    double scale=std::min((width-28)/std::max(.01,xmax-xmin),(height-78)/std::max(.01,ymax-ymin));
    auto screen=[&](xy p)->xy{return {x+width*.5+(p.x-(xmin+xmax)*.5)*scale,y+43+(height-78)*.5+(p.y-(ymin+ymax)*.5)*scale};};
    out<<"<g id=\"global-locator\" data-total-components=\""<<components.size()<<"\">"
       <<"<rect x=\""<<x<<"\" y=\""<<y<<"\" width=\""<<width<<"\" height=\""<<height<<"\" rx=\"7\" fill=\"#eef2f5\" stroke=\"#b5c3cd\"/>"
       <<"<text x=\""<<x+10<<"\" y=\""<<y+20<<"\" font-size=\"13\">Global locator | rough isometric</text>";
    std::size_t stride=std::max<std::size_t>(1,(global["wires"].size()+399)/400),drawn=0;
    std::ostringstream wire_path;
    for(std::size_t i=0;i<global["wires"].size();i+=stride)
    {
        auto const& w=global["wires"][i];
        auto a=projected.find(w["Source"].get<std::string>()),b=projected.find(w["Target"].get<std::string>());
        if(a==projected.end()||b==projected.end())continue;
        auto p=screen(a->second),q=screen(b->second);
        wire_path<<"M "<<p.x<<' '<<p.y<<" L "<<q.x<<' '<<q.y<<' ';
        ++drawn;
    }
    out<<"<path d=\""<<wire_path.str()<<"\" fill=\"none\" stroke=\"#b4c1ca\" opacity=\".5\" stroke-width=\".55\"/>";
    json selected_ids=json::array(),selected_refs=json::array();
    json markers=json::array();
    // The selected markers are drawn last so dense clusters remain visible.
    for(bool selected:{false,true})
    {
        std::ostringstream marker_path;
        double r=selected?2.7:1.5;
        for(auto const& c:components)
        {
            auto id=c["id"].get<std::string>();
            if(visible.contains(id)!=selected)continue;
            auto p=screen(projected.at(id));
            marker_path<<"M "<<p.x-r<<' '<<p.y<<" a "<<r<<' '<<r<<" 0 1 0 "<<2*r<<" 0 a "<<r<<' '<<r<<" 0 1 0 "<<-2*r<<" 0 Z ";
            markers.push_back({{"id",id},{"ref",c["ref"]},{"x",p.x},{"y",p.y},{"selected",selected}});
            if(selected){selected_ids.push_back(id);selected_refs.push_back(c["ref"]);}
        }
        out<<"<path data-mini-layer=\""<<(selected?"selected":"unselected")<<"\" d=\""<<marker_path.str()
           <<"\" fill=\""<<(selected?"#ffd33d":"#8399aa")<<"\" stroke=\""<<(selected?"#9a7000":"#678296")<<"\" stroke-width=\".45\"/>";
    }
    // One metadata node retains auditable component-to-marker ownership while
    // only three batched SVG paths reach Cairo, even for thousands of gates.
    out<<"<metadata id=\"locator-markers\">"<<escaped(markers.dump())<<"</metadata>";
    out<<"<text x=\""<<x+10<<"\" y=\""<<y+height-17<<"\" font-size=\"12\">Yellow: in main view "<<visible.size()<<'/'<<components.size()<<"</text></g>";
    return {{"enabled",true},{"total_components",components.size()},{"highlighted_components",selected_ids.size()},
            {"highlighted_ids",selected_ids},{"highlighted_refs",selected_refs},{"wire_total",global["wires"].size()},{"wire_drawn",drawn},
            {"projection","fixed-isometric-all-original-centers"},{"rough",true},{"geometry_mutated",false},
            {"bounds",{x,y,width,height}},{"note","Locator includes all original components and distant outliers. Yellow means visible in the main view; center-to-center wires may be sampled, not pin-level evidence."}};
}

std::string schematic_pin_label(json const& component, int pin)
{
    auto const& native = component.value("native", json::object());
    if (native.contains("pin_labels") && native["pin_labels"].is_array() && pin >= 0 &&
        static_cast<std::size_t>(pin) < native["pin_labels"].size() && native["pin_labels"][pin].is_string())
        return native["pin_labels"][pin].get<std::string>();
    static std::map<std::string, std::vector<std::string>> const labels{
        {"Ground Component", {"gnd"}}, {"Logic Input", {"out"}}, {"Logic Output", {"in"}},
        {"Resistor", {"1", "2"}}, {"Basic Capacitor", {"+", "-"}},
        {"Basic Inductor", {"1", "2"}}, {"Battery Source", {"+", "-"}},
        {"Current Source", {"+", "-"}}, {"Sinewave Source", {"+", "-"}},
        {"Square Source", {"+", "-"}}, {"Triangle Source", {"+", "-"}},
        {"Sawtooth Source", {"+", "-"}}, {"Pulse Source", {"+", "-"}},
        {"Basic Diode", {"anode", "cathode"}}, {"Light-Emitting Diode", {"anode", "cathode"}},
        {"Photodiode", {"anode", "cathode"}}, {"Photoresistor", {"1", "2"}},
        {"Buzzer", {"+", "-"}}, {"Spark Gap", {"1", "2"}}, {"Tesla Coil", {"1", "2"}},
        {"Dual Light-Emitting Diode", {"1", "2"}}, {"Electric Bell", {"+", "-"}},
        {"Electric Fan", {"+", "-"}}, {"Musical Box", {"+", "-"}},
        {"Incandescent Lamp", {"1", "2"}}, {"Fuse Component", {"1", "2"}},
        {"Resistance Box", {"1", "2"}}, {"Multimeter", {"+", "-"}},
        {"Simple Switch", {"1", "2"}}, {"Push Switch", {"1", "2"}},
        {"Air Switch", {"1", "2"}}, {"SPDT Switch", {"left", "common", "right"}},
        {"DPDT Switch", {"left-low", "common-low", "right-low", "left-high", "common-high", "right-high"}},
        {"Slide Rheostat", {"left-wiper", "right-wiper", "left-end", "right-end"}},
        {"Student Source", {"left", "left-mid", "right-mid", "right"}},
        {"Analog Joystick", {"x-end-1", "x-wiper", "x-end-2", "y-end-1", "y-wiper", "y-end-2"}},
        {"Color Light-Emitting Diode", {"left-up", "left-mid", "left-low", "right"}},
        {"Resistance Law", {"left-0", "left-1", "left-2", "left-3", "right-0", "right-1", "right-2", "right-3"}},
        {"Solenoid", {"sub+", "sub-", "+", "-"}},
        {"Simple Instrument", {"in", "out"}}, {"Simple Ammeter", {"left", "common", "right"}},
        {"Simple Voltmeter", {"left", "common", "right"}}, {"Galvanometer", {"left", "common", "right"}},
        {"Microammeter", {"left", "common", "right"}},
        {"Electricity Meter", {"left", "right-mid", "left-mid", "right"}},
        {"Proximity Sensor", {"out"}},
        {"Accelerometer", {"x", "y", "z"}}, {"Attitude Sensor", {"x", "y", "z"}},
        {"Gravity Sensor", {"x", "y", "z"}}, {"Gyroscope", {"x", "y", "z"}},
        {"Linear Accelerometer", {"x", "y", "z"}}, {"Magnetic Field Sensor", {"x", "y", "z"}},
        {"Operational Amplifier", {"in-", "in+", "out"}},
        {"Comparator", {"out", "in+", "in-"}}, {"Transistor", {"base", "collector", "emitter"}},
        {"N-MOSFET", {"gate", "source", "drain"}}, {"P-MOSFET", {"gate", "drain", "source"}},
        // Native PhysicsLab order alternates the left/right winding ends.
        {"Transformer", {"primary+", "secondary+", "primary-", "secondary-"}},
        {"Tapped Transformer", {"primary+", "secondary+", "primary-", "secondary-", "center-tap"}},
        {"Mutual Inductor", {"L1+", "L2+", "L1-", "L2-"}},
        {"Rectifier", {"left-up", "right-up", "left-low", "right-low"}},
        {"Relay Component", {"left-up", "common", "left-low", "coil+", "coil-"}},
        {"555 Timer", {"VCC", "DIS", "THR", "CTRL", "TRIG", "OUT", "RESET", "GND"}},
        {"Yes Gate", {"in", "out"}}, {"No Gate", {"in", "out"}},
        {"And Gate", {"in-1", "in-2", "out"}}, {"Or Gate", {"in-1", "in-2", "out"}},
        {"Xor Gate", {"in-1", "in-2", "out"}}, {"Xnor Gate", {"in-1", "in-2", "out"}},
        {"Nand Gate", {"in-1", "in-2", "out"}}, {"Nor Gate", {"in-1", "in-2", "out"}},
        {"Imp Gate", {"in-1", "in-2", "out"}}, {"Nimp Gate", {"in-1", "in-2", "out"}},
        {"Half Adder", {"sum", "carry", "B", "A"}},
        {"Full Adder", {"sum", "carry-out", "B", "carry-in", "A"}},
        {"Half Subtractor", {"difference", "borrow", "B", "A"}},
        {"Full Subtractor", {"difference", "borrow-out", "B", "borrow-in", "A"}},
        {"Multiplier", {"p3", "p2", "p1", "p0", "b1", "b0", "a1", "a0"}},
        {"D Flipflop", {"q", "q_bar", "d", "clk"}},
        {"T Flipflop", {"q", "q_bar", "t", "clk"}},
        {"Real-T Flipflop", {"q", "q_bar", "t", "clk"}},
        {"JK Flipflop", {"q", "q_bar", "j", "clk", "k"}},
        {"Schmitt Trigger", {"in", "out"}},
        {"Random Generator", {"q3", "q2", "q1", "q0", "clk", "reset_n"}},
        {"Counter", {"q3", "q2", "q1", "q0", "clk", "enable"}},
        {"8bit Input", {"i3", "i2", "i1", "i0", "o3", "o2", "o1", "o0"}},
        {"8bit Display", {"i3", "i2", "i1", "i0", "o3", "o2", "o1", "o0"}},
        {"Eight Bit Display", {"i3", "i2", "i1", "i0", "o3", "o2", "o1", "o0"}},
    };
    auto it = labels.find(component.at("type").get<std::string>());
    if (it != labels.end() && pin >= 0 && static_cast<std::size_t>(pin) < it->second.size()) return it->second[pin];
    return "p" + std::to_string(pin);
}

std::string schematic_symbol(std::string type)
{
    static std::map<std::string, std::string> const names{
        {"Resistor", "R"}, {"Basic Capacitor", "C"}, {"Basic Inductor", "L"},
        {"Battery Source", "VDC"}, {"Current Source", "IDC"}, {"Sinewave Source", "VAC"},
        {"Square Source", "SQUARE"}, {"Triangle Source", "TRI"}, {"Sawtooth Source", "SAW"},
        {"Pulse Source", "PULSE"}, {"Operational Amplifier", "OP AMP"},
        {"Comparator", "CMP"}, {"Random Generator", "LFSR4"}, {"Ground Component", "GND"},
        {"Logic Input", "IN"}, {"Logic Output", "OUT"}, {"555 Timer", "555"},
        {"Slide Rheostat", "R VAR"}, {"Relay Component", "RELAY"}, {"Rectifier", "BRIDGE"},
        {"Student Source", "AC/DC"},
    };
    if (names.contains(type)) return names.at(type);
    if (type.ends_with(" Gate"))
    {
        type.resize(type.size() - 5);
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
        return type == "NO" ? "NOT" : type;
    }
    return utf8_ellipsize(type, 17);
}

// A position-aware electrical schematic. Native coordinates determine the
// rank/order of non-overlapping component cells; connectivity always comes
// from exact saved/native node membership. This is a read-only projection,
// never a rewritten layout or an electrical inference.
json render_schematic(json const& data, json const& global, std::ostream& out,
                      std::size_t offset, std::size_t limit)
{
    auto const& components = data.at("components");
    auto end = std::min(components.size(), offset + limit);
    if (offset >= components.size()) throw std::runtime_error("page offset exceeds component count");
    std::size_t count = end - offset;
    if (!count || count > 24) throw std::runtime_error("schematic view requires 1..24 selected components");
    constexpr double canvas_width = 1500, canvas_height = 1040;
    constexpr double left = 36, top = 112, scene_width = 1100, scene_height = 850;
    std::size_t cols = std::min<std::size_t>(6, std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(count) * 1.45)))));
    std::size_t rows = (count + cols - 1) / cols;
    double cell_w = scene_width / cols, cell_h = scene_height / rows;
    double box_w = std::min(160.0, cell_w - 28);

    struct box { std::size_t component_index{}; double x{}, y{}, w{}, h{}; std::map<int, xy> pins; };
    std::vector<std::size_t> order;
    for (std::size_t i = offset; i < end; ++i) order.push_back(i);
    auto projected = [&](json const& c) {
        auto const& p = c.at("position");
        return xy{p[0].get<double>() - p[1].get<double>() * .18,
                  p[1].get<double>() + p[0].get<double>() * .08 - p[2].get<double>() * .22};
    };
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        auto p = projected(components[a]), q = projected(components[b]);
        if (std::abs(p.y - q.y) > 1e-12) return p.y < q.y;
        if (std::abs(p.x - q.x) > 1e-12) return p.x < q.x;
        return components[a].at("id").template get<std::string>() < components[b].at("id").template get<std::string>();
    });
    // Preserve vertical position bands, then restore native left-to-right
    // order inside each band. Rank-based placement is robust to extreme saved
    // outliers and guarantees labels never cover component bodies.
    for (std::size_t row = 0; row < rows; ++row)
    {
        auto first = order.begin() + static_cast<std::ptrdiff_t>(row * cols);
        auto last = order.begin() + static_cast<std::ptrdiff_t>(std::min(count, (row + 1) * cols));
        std::stable_sort(first, last, [&](auto a, auto b) {
            auto p = projected(components[a]), q = projected(components[b]);
            if (std::abs(p.x - q.x) > 1e-12) return p.x < q.x;
            return components[a].at("id").template get<std::string>() < components[b].at("id").template get<std::string>();
        });
    }
    std::map<std::string, box> boxes;
    for (std::size_t rank = 0; rank < order.size(); ++rank)
    {
        auto index = order[rank];
        auto const& c = components[index];
        std::size_t pin_count = c.at("pins").size();
        std::size_t side = std::max<std::size_t>(1, (pin_count + 1) / 2);
        double h = pin_count <= 2 ? 76.0 : std::min(cell_h - 24, std::max(94.0, 38.0 + side * 20.0));
        box b{index, left + (rank % cols) * cell_w + (cell_w - box_w) / 2,
              top + (rank / cols) * cell_h + (cell_h - h) / 2, box_w, h, {}};
        auto type = c.at("type").get<std::string>();
        std::size_t ordinal = 0;
        for (auto const& pin : c.at("pins"))
        {
            int number = pin.at("pin").get<int>();
            xy point{};
            if (pin_count == 1)
            {
                if (type == "Ground Component") point = {b.x + b.w / 2, b.y + 14};
                else if (type == "Logic Input") point = {b.x + b.w, b.y + b.h / 2};
                else point = {b.x, b.y + b.h / 2};
            }
            else if (pin_count == 2) point = {ordinal == 0 ? b.x : b.x + b.w, b.y + b.h / 2};
            else if (type == "Operational Amplifier" && pin_count == 3)
                point = number == 0 ? xy{b.x, b.y + b.h * .36} : number == 1 ? xy{b.x, b.y + b.h * .68} : xy{b.x + b.w, b.y + b.h * .52};
            else if (type == "Comparator" && pin_count == 3)
                point = number == 0 ? xy{b.x + b.w, b.y + b.h * .52} : number == 1 ? xy{b.x, b.y + b.h * .36} : xy{b.x, b.y + b.h * .68};
            else if ((type == "Transistor" || type == "N-MOSFET" || type == "P-MOSFET") && pin_count == 3)
                point = number == 0 ? xy{b.x, b.y + b.h * .52}
                      : ((type == "N-MOSFET") == (number == 1)) ? xy{b.x + b.w, b.y + b.h * .74}
                                                                 : xy{b.x + b.w, b.y + b.h * .30};
            else if ((type == "Rectifier" || type == "Slide Rheostat" || type == "Electricity Meter") && pin_count == 4)
                point = number % 2 == 0 ? xy{b.x, b.y + b.h * (number == 0 ? .34 : .68)}
                                        : xy{b.x + b.w, b.y + b.h * (number == 1 ? .34 : .68)};
            else if (type == "Color Light-Emitting Diode" && pin_count == 4)
                point = number == 3 ? xy{b.x + b.w, b.y + b.h * .50}
                                    : xy{b.x, b.y + b.h * (.25 + number * .25)};
            else if (type == "SPDT Switch" && pin_count == 3)
                point = number == 1 ? xy{b.x, b.y + b.h * .50}
                                    : xy{b.x + b.w, b.y + b.h * (number == 0 ? .32 : .68)};
            else if (type == "DPDT Switch" && pin_count == 6)
            {
                bool common = number == 1 || number == 4;
                bool upper_pole = number < 3;
                std::size_t throw_index = static_cast<std::size_t>(number % 3 == 0 ? 0 : 1);
                point = common ? xy{b.x, b.y + b.h * (upper_pole ? .32 : .68)}
                               : xy{b.x + b.w, b.y + b.h * (upper_pole ? (.24 + throw_index*.16) : (.60 + throw_index*.16))};
            }
            else if ((type == "Transformer" || type == "Mutual Inductor") && pin_count == 4)
                point = number == 0 || number == 2 ? xy{b.x, b.y + b.h * (number == 0 ? .30 : .70)}
                                                   : xy{b.x + b.w, b.y + b.h * (number == 1 ? .30 : .70)};
            else if (type == "Tapped Transformer" && pin_count == 5)
                point = number == 0 || number == 2 ? xy{b.x, b.y + b.h * (number == 0 ? .30 : .70)}
                                                   : xy{b.x + b.w, b.y + b.h * (number == 1 ? .24 : number == 4 ? .50 : .76)};
            else if (type.ends_with(" Gate") && pin_count == 3)
                point = number == 2 ? xy{b.x + b.w, b.y + b.h * .50}
                                    : xy{b.x, b.y + b.h * (number == 0 ? .35 : .65)};
            else if ((type == "Half Adder" || type == "Full Adder" || type == "Half Subtractor" ||
                      type == "Full Subtractor" || type == "D Flipflop" || type == "T Flipflop" ||
                      type == "Real-T Flipflop" || type == "JK Flipflop") && pin_count >= 4)
            {
                bool output = number < 2;
                std::size_t local = output ? static_cast<std::size_t>(number) : static_cast<std::size_t>(number - 2);
                std::size_t local_count = output ? 2 : pin_count - 2;
                point = {output ? b.x + b.w : b.x,
                         b.y + 16 + (local + 1) * (b.h - 32) / (local_count + 1)};
            }
            else if ((type == "Multiplier" || type == "Counter" || type == "Random Generator") && pin_count >= 6)
            {
                bool output = number < 4;
                std::size_t local = output ? static_cast<std::size_t>(number) : static_cast<std::size_t>(number - 4);
                std::size_t local_count = output ? 4 : pin_count - 4;
                point = {output ? b.x + b.w : b.x,
                         b.y + 16 + (local + 1) * (b.h - 32) / (local_count + 1)};
            }
            else if ((type == "8bit Input" || type == "8bit Display" || type == "Eight Bit Display") && pin_count == 8)
            {
                bool output = number >= 4;
                std::size_t local = static_cast<std::size_t>(output ? number - 4 : number);
                point = {output ? b.x + b.w : b.x,
                         b.y + 16 + (local + 1) * (b.h - 32) / 5};
            }
            else
            {
                bool on_left = ordinal < (pin_count + 1) / 2;
                std::size_t local = on_left ? ordinal : ordinal - (pin_count + 1) / 2;
                std::size_t local_count = on_left ? (pin_count + 1) / 2 : pin_count / 2;
                point = {on_left ? b.x : b.x + b.w,
                         b.y + 16 + (local + 1) * (b.h - 32) / (local_count + 1)};
            }
            b.pins.emplace(number, point);
            ++ordinal;
        }
        boxes.emplace(c.at("id").get<std::string>(), std::move(b));
    }

    struct terminal { std::string component; int pin{}; xy point{}; };
    std::map<std::string, std::vector<terminal>> node_terms;
    for (auto const& [id, b] : boxes)
    {
        auto const& c = components[b.component_index];
        for (auto const& pin : c.at("pins"))
        {
            int number = pin.at("pin").get<int>();
            node_terms[pin.at("node").get<std::string>()].push_back({id, number, b.pins.at(number)});
        }
    }
    std::map<std::string, std::size_t> global_connections;
    for (auto const& node : global.at("nodes")) global_connections[node.at("id").get<std::string>()] = node.at("connections").size();

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << canvas_width << "\" height=\"" << canvas_height
        << "\" viewBox=\"0 0 " << canvas_width << ' ' << canvas_height << "\">"
        << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/><g font-family=\"DejaVu Sans,sans-serif\" fill=\"#111827\">";
    auto text = [&](double x, double y, std::string const& value, int size = 13, char const* fill = "#132531") {
        out << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"" << size << "\" fill=\"" << fill << "\">" << escaped(value) << "</text>";
    };
    text(28, 34, "Aurex electrical schematic", 23);
    text(28, 60, "Auto-layout follows saved component positions; orthogonal wires follow exact electrical nodes.", 13);
    text(28, 81, "Filled dot = true shared junction. Crossing lines without a dot are not connected. Open X = unconnected.", 12);
    out << "<defs><marker id=\"external-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#596b78\"/></marker></defs>";
    out << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << scene_width << "\" height=\"" << scene_height
        << "\" fill=\"none\" stroke=\"#d8dee3\" stroke-width=\"1\"/>";
    std::size_t node_index = 0, junctions = 0, unconnected = 0, external = 0, routed = 0;
    for (auto const& [node, terms] : node_terms)
    {
        std::size_t total = global_connections.contains(node) ? global_connections.at(node) : terms.size();
        ++node_index;
        char const* color = "#20262b";
        if (terms.size() == 1)
        {
            auto p = terms.front().point;
            if (total <= 1)
            {
                ++unconnected;
                out << "<path data-node=\"" << escaped(node) << "\" d=\"M " << p.x-5 << ' ' << p.y-5 << " L " << p.x+5 << ' ' << p.y+5
                    << " M " << p.x-5 << ' ' << p.y+5 << " L " << p.x+5 << ' ' << p.y-5 << "\" stroke=\"#b23a3a\" stroke-width=\"1.8\"/>";
            }
            else
            {
                ++external;
                bool go_left = p.x < left + scene_width / 2;
                double qx = p.x + (go_left ? -34 : 34);
                out << "<path data-node=\"" << escaped(node) << "\" d=\"M " << p.x << ' ' << p.y << " H " << qx
                    << "\" stroke=\"" << color << "\" stroke-width=\"2\" marker-end=\"url(#external-arrow)\"/>";
                text(go_left ? qx - 6 - node.size()*7 : qx + 5, p.y - 5, node + " ext(" + std::to_string(total-1) + ")", 10, "#596b78");
            }
            continue;
        }
        if (terms.size() == 2)
        {
            auto p = terms[0].point, q = terms[1].point;
            double channel_x = (p.x + q.x) / 2 + (static_cast<int>(node_index % 5) - 2) * 5;
            out << "<path data-node=\"" << escaped(node) << "\" d=\"M " << p.x << ' ' << p.y << " H " << channel_x
                << " V " << q.y << " H " << q.x << "\" fill=\"none\" stroke=\"#20262b\" stroke-width=\"1.8\"/>";
            ++routed;
            if (node_terms.size() <= 18) text(channel_x + 4, (p.y + q.y) / 2 - 5, node, 9, "#35627d");
            continue;
        }
        std::vector<double> xs, ys;
        for (auto const& term : terms) { xs.push_back(term.point.x); ys.push_back(term.point.y); }
        std::sort(xs.begin(), xs.end()); std::sort(ys.begin(), ys.end());
        double hub_x = xs[xs.size()/2] + (static_cast<int>(node_index % 5) - 2) * 6;
        double hub_y = ys[ys.size()/2] + (static_cast<int>((node_index/5) % 5) - 2) * 6;
        hub_x = std::clamp(hub_x, left + 12, left + scene_width - 12);
        hub_y = std::clamp(hub_y, top + 12, top + scene_height - 12);
        for (auto const& term : terms)
            out << "<path data-node=\"" << escaped(node) << "\" d=\"M " << term.point.x << ' ' << term.point.y << " H " << hub_x << " V " << hub_y
                << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2.1\" opacity=\".9\"/>";
        ++routed;
        if (total >= 3)
        {
            ++junctions;
            out << "<circle data-junction-node=\"" << escaped(node) << "\" cx=\"" << hub_x << "\" cy=\"" << hub_y << "\" r=\"4\" fill=\"" << color << "\"/>";
        }
        if (node_terms.size() <= 18 || total >= 3)
        {
            double label_w = std::max(24.0, node.size() * 6.4 + 8);
            out << "<rect x=\"" << hub_x + 5 << "\" y=\"" << hub_y - 15 << "\" width=\"" << label_w << "\" height=\"15\" rx=\"2\" fill=\"#ffffff\" fill-opacity=\".9\"/>";
            text(hub_x + 9, hub_y - 4, node, 10, "#35627d");
        }
    }
    // Conventional schematic symbols cover any route passing behind them.
    // Multi-pin and uncommon PhysicsLab devices use IEC-style functional
    // blocks with exact pin names; this is preferable to inventing a symbol.
    for (auto const& [id, b] : boxes)
    {
        auto const& c = components[b.component_index];
        auto type = c.at("type").get<std::string>();
        auto ref = c.at("ref").get<std::string>();
        auto cx = b.x + b.w / 2, cy = b.y + b.h / 2;
        out << "<g data-component-id=\"" << escaped(id) << "\" data-native-position=\"" << escaped(c.at("position").dump()) << "\">";
        auto stroke = [&](std::string const& path, double width = 1.8) {
            out << "<path d=\"" << path << "\" fill=\"none\" stroke=\"#111827\" stroke-width=\"" << width << "\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>";
        };
        auto component_title = ref + " " + schematic_symbol(type);
        auto label = c.value("label", "");
        if (!label.empty())
        {
            label = utf8_ellipsize(label, 16);
            component_title += " " + label;
        }
        out << "<rect x=\"" << b.x-2 << "\" y=\"" << b.y-23 << "\" width=\"" << b.w+4
            << "\" height=\"17\" fill=\"white\"/>";
        text(b.x + 3, b.y - 10, component_title, 12, "#111827");
        auto properties = compact_properties(c);
        if (!properties.empty())
        {
            properties = utf8_ellipsize(properties, 28);
            out << "<rect x=\"" << b.x-2 << "\" y=\"" << b.y+b.h+4 << "\" width=\"" << b.w+4
                << "\" height=\"17\" fill=\"white\"/>";
            text(b.x + 3, b.y + b.h + 17, properties, 10, "#374151");
        }
        if (type == "Ground Component")
        {
            auto p = b.pins.begin()->second;
            stroke("M " + std::to_string(p.x) + " " + std::to_string(p.y) + " V " + std::to_string(p.y+15));
            stroke("M " + std::to_string(p.x-15) + " " + std::to_string(p.y+15) + " H " + std::to_string(p.x+15));
            stroke("M " + std::to_string(p.x-10) + " " + std::to_string(p.y+21) + " H " + std::to_string(p.x+10));
            stroke("M " + std::to_string(p.x-5) + " " + std::to_string(p.y+27) + " H " + std::to_string(p.x+5));
        }
        else if (c.at("pins").size() == 2)
        {
            auto a = b.pins.at(c.at("pins")[0].at("pin").get<int>());
            auto z = b.pins.at(c.at("pins")[1].at("pin").get<int>());
            double x1 = a.x + 31, x2 = z.x - 31;
            out << "<rect x=\"" << x1-4 << "\" y=\"" << cy-30 << "\" width=\"" << x2-x1+8
                << "\" height=\"60\" fill=\"white\"/>";
            if (type == "Resistor" || type == "Resistance Box" || type == "Photoresistor" || type == "Slide Rheostat")
            {
                std::ostringstream path; path << "M " << a.x << ' ' << cy << " H " << x1;
                for (int i = 0; i < 8; ++i) path << " L " << x1 + (i+1)*(x2-x1)/9 << ' ' << cy + (i%2 ? -10 : 10);
                path << " L " << x2 << ' ' << cy << " H " << z.x; stroke(path.str());
                if (type == "Slide Rheostat") stroke("M " + std::to_string(cx+18) + " " + std::to_string(cy-24) + " L " + std::to_string(cx+2) + " " + std::to_string(cy-5));
            }
            else if (type == "Basic Capacitor")
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-7));
                stroke("M " + std::to_string(cx-7) + " " + std::to_string(cy-20) + " V " + std::to_string(cy+20), 2.4);
                stroke("M " + std::to_string(cx+7) + " " + std::to_string(cy-20) + " V " + std::to_string(cy+20), 2.4);
                stroke("M " + std::to_string(cx+7) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
            }
            else if (type == "Basic Inductor")
            {
                std::ostringstream path; path << "M " << a.x << ' ' << cy << " H " << x1;
                double step=(x2-x1)/4;
                for(int i=0;i<4;++i) path << " C " << x1+(i+.15)*step << ' ' << cy-18 << ' ' << x1+(i+.85)*step << ' ' << cy-18 << ' ' << x1+(i+1)*step << ' ' << cy;
                path << " H " << z.x; stroke(path.str());
            }
            else if (type == "Battery Source")
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-8));
                stroke("M " + std::to_string(cx-8) + " " + std::to_string(cy-22) + " V " + std::to_string(cy+22), 2.5);
                stroke("M " + std::to_string(cx+8) + " " + std::to_string(cy-13) + " V " + std::to_string(cy+13), 2.5);
                stroke("M " + std::to_string(cx+8) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                text(cx-22,cy-23,"+",13); text(cx+14,cy-13,"-",13);
            }
            else if (type == "Current Source" || type.ends_with(" Source") || type == "Student Source")
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-25));
                stroke("M " + std::to_string(cx+25) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                out << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"25\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
                if (type == "Current Source") { stroke("M " + std::to_string(cx) + " " + std::to_string(cy+13) + " V " + std::to_string(cy-13)); stroke("M " + std::to_string(cx) + " " + std::to_string(cy-13) + " l -5 8 M " + std::to_string(cx) + " " + std::to_string(cy-13) + " l 5 8"); }
                else { std::ostringstream wave; wave << "M " << cx-15 << ' ' << cy << " C " << cx-10 << ' ' << cy-12 << ' ' << cx-5 << ' ' << cy-12 << ' ' << cx << ' ' << cy << " C " << cx+5 << ' ' << cy+12 << ' ' << cx+10 << ' ' << cy+12 << ' ' << cx+15 << ' ' << cy; stroke(wave.str(),1.4); }
            }
            else if (type == "Simple Switch" || type == "Push Switch" || type == "Air Switch")
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-24));
                stroke("M " + std::to_string(cx+24) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                out << "<circle cx=\"" << cx-24 << "\" cy=\"" << cy << "\" r=\"3\" fill=\"#111827\"/><circle cx=\"" << cx+24 << "\" cy=\"" << cy << "\" r=\"3\" fill=\"#111827\"/>";
                stroke("M " + std::to_string(cx-22) + " " + std::to_string(cy-2) + " L " + std::to_string(cx+17) + " " + std::to_string(cy-19),2.1);
            }
            else if (type.find("Diode") != std::string::npos)
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-18));
                out << "<polygon points=\"" << cx-18 << ',' << cy-16 << ' ' << cx+10 << ',' << cy << ' ' << cx-18 << ',' << cy+16 << "\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
                stroke("M " + std::to_string(cx+10) + " " + std::to_string(cy-18) + " V " + std::to_string(cy+18),2.2);
                stroke("M " + std::to_string(cx+10) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                if (type.find("Light") != std::string::npos) { stroke("M " + std::to_string(cx+14) + " " + std::to_string(cy-18) + " l 12 -12"); stroke("M " + std::to_string(cx+23) + " " + std::to_string(cy-10) + " l 12 -12"); }
            }
            else if (type == "Incandescent Lamp" || type == "Electric Bell" || type == "Electric Fan" || type == "Musical Box")
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-25));
                stroke("M " + std::to_string(cx+25) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                out << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"25\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
                if (type == "Incandescent Lamp") stroke("M " + std::to_string(cx-16) + " " + std::to_string(cy-16) + " L " + std::to_string(cx+16) + " " + std::to_string(cy+16) + " M " + std::to_string(cx-16) + " " + std::to_string(cy+16) + " L " + std::to_string(cx+16) + " " + std::to_string(cy-16));
                else text(cx-13,cy+6,schematic_symbol(type),11);
            }
            else
            {
                stroke("M " + std::to_string(a.x) + " " + std::to_string(cy) + " H " + std::to_string(cx-28));
                out << "<rect x=\"" << cx-28 << "\" y=\"" << cy-18 << "\" width=\"56\" height=\"36\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
                stroke("M " + std::to_string(cx+28) + " " + std::to_string(cy) + " H " + std::to_string(z.x));
                auto symbol=schematic_symbol(type); text(cx-symbol.size()*3.2,cy+4,symbol,10);
            }
        }
        else if (type == "SPDT Switch" && c.at("pins").size() == 3)
        {
            auto upper=b.pins.at(0),common=b.pins.at(1),lower=b.pins.at(2);
            out << "<rect x=\"" << cx-39 << "\" y=\"" << cy-36 << "\" width=\"78\" height=\"72\" fill=\"white\"/>";
            stroke("M "+std::to_string(common.x)+" "+std::to_string(common.y)+" H "+std::to_string(cx-20));
            stroke("M "+std::to_string(cx+20)+" "+std::to_string(upper.y)+" H "+std::to_string(upper.x));
            stroke("M "+std::to_string(cx+20)+" "+std::to_string(lower.y)+" H "+std::to_string(lower.x));
            out << "<circle cx=\"" << cx-20 << "\" cy=\"" << common.y << "\" r=\"3\" fill=\"#111827\"/>"
                << "<circle cx=\"" << cx+20 << "\" cy=\"" << upper.y << "\" r=\"3\" fill=\"#111827\"/>"
                << "<circle cx=\"" << cx+20 << "\" cy=\"" << lower.y << "\" r=\"3\" fill=\"#111827\"/>";
            stroke("M "+std::to_string(cx-18)+" "+std::to_string(common.y-2)+" L "+std::to_string(cx+15)+" "+std::to_string(upper.y));
        }
        else if (type == "DPDT Switch" && c.at("pins").size() == 6)
        {
            out << "<rect x=\"" << cx-43 << "\" y=\"" << cy-48 << "\" width=\"86\" height=\"96\" fill=\"white\"/>";
            for (int pole=0;pole<2;++pole)
            {
                int base=pole*3;auto a=b.pins.at(base),common=b.pins.at(base+1),z=b.pins.at(base+2);
                stroke("M "+std::to_string(common.x)+" "+std::to_string(common.y)+" H "+std::to_string(cx-20));
                stroke("M "+std::to_string(cx+20)+" "+std::to_string(a.y)+" H "+std::to_string(a.x));
                stroke("M "+std::to_string(cx+20)+" "+std::to_string(z.y)+" H "+std::to_string(z.x));
                out << "<circle cx=\"" << cx-20 << "\" cy=\"" << common.y << "\" r=\"2.8\" fill=\"#111827\"/>"
                    << "<circle cx=\"" << cx+20 << "\" cy=\"" << a.y << "\" r=\"2.8\" fill=\"#111827\"/>"
                    << "<circle cx=\"" << cx+20 << "\" cy=\"" << z.y << "\" r=\"2.8\" fill=\"#111827\"/>";
                stroke("M "+std::to_string(cx-18)+" "+std::to_string(common.y-2)+" L "+std::to_string(cx+15)+" "+std::to_string(a.y));
            }
        }
        else if (type == "Slide Rheostat" && c.at("pins").size() == 4)
        {
            auto lw=b.pins.at(0),rw=b.pins.at(1),le=b.pins.at(2),re=b.pins.at(3);
            out << "<rect x=\"" << cx-48 << "\" y=\"" << cy-38 << "\" width=\"96\" height=\"76\" fill=\"white\"/>";
            double ry=cy+13,x1=cx-32,x2=cx+32;
            stroke("M "+std::to_string(le.x)+" "+std::to_string(le.y)+" H "+std::to_string(x1)+" V "+std::to_string(ry));
            std::ostringstream path;path<<"M "<<x1<<' '<<ry;
            for(int i=0;i<8;++i)path<<" L "<<x1+(i+1)*(x2-x1)/9<<' '<<ry+(i%2?-8:8);
            path<<" L "<<x2<<' '<<ry;stroke(path.str());
            stroke("M "+std::to_string(x2)+" "+std::to_string(ry)+" V "+std::to_string(re.y)+" H "+std::to_string(re.x));
            stroke("M "+std::to_string(lw.x)+" "+std::to_string(lw.y)+" H "+std::to_string(cx-18)+" L "+std::to_string(cx-8)+" "+std::to_string(ry-4));
            stroke("M "+std::to_string(rw.x)+" "+std::to_string(rw.y)+" H "+std::to_string(cx+18)+" L "+std::to_string(cx+8)+" "+std::to_string(ry-4));
        }
        else if ((type == "Operational Amplifier" || type == "Comparator") && c.at("pins").size() == 3)
        {
            auto neg=b.pins.at(type=="Comparator"?2:0),pos=b.pins.at(1),output=b.pins.at(type=="Comparator"?0:2);
            out << "<polygon points=\"" << b.x+22 << ',' << b.y+12 << ' ' << b.x+22 << ',' << b.y+b.h-12 << ' ' << b.x+b.w-18 << ',' << cy << "\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
            stroke("M "+std::to_string(neg.x)+" "+std::to_string(neg.y)+" H "+std::to_string(b.x+22));
            stroke("M "+std::to_string(pos.x)+" "+std::to_string(pos.y)+" H "+std::to_string(b.x+22));
            stroke("M "+std::to_string(b.x+b.w-18)+" "+std::to_string(cy)+" H "+std::to_string(output.x));
            text(b.x+27,neg.y+4,"-",13);text(b.x+27,pos.y+4,"+",13);
        }
        else if ((type == "Transistor" || type == "N-MOSFET" || type == "P-MOSFET") && c.at("pins").size() == 3)
        {
            auto base=b.pins.at(0),upper=b.pins.at(1),lower=b.pins.at(2);
            out << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"31\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.5\"/>";
            stroke("M "+std::to_string(base.x)+" "+std::to_string(base.y)+" H "+std::to_string(cx-12));
            stroke("M "+std::to_string(cx-12)+" "+std::to_string(cy-20)+" V "+std::to_string(cy+20),2.2);
            stroke("M "+std::to_string(cx-12)+" "+std::to_string(cy-12)+" L "+std::to_string(upper.x)+" "+std::to_string(upper.y));
            stroke("M "+std::to_string(cx-12)+" "+std::to_string(cy+12)+" L "+std::to_string(lower.x)+" "+std::to_string(lower.y));
            text(cx-4,cy+5,type=="Transistor"?"BJT":(type=="N-MOSFET"?"NMOS":"PMOS"),9);
        }
        else if ((type == "Transformer" || type == "Mutual Inductor" || type == "Tapped Transformer") &&
                 (c.at("pins").size() == 4 || c.at("pins").size() == 5))
        {
            out << "<rect x=\"" << cx-39 << "\" y=\"" << cy-39 << "\" width=\"78\" height=\"78\" fill=\"white\"/>";
            auto coil = [&](double x, double y0, double y1, bool face_right) {
                std::ostringstream path; path << "M " << x << ' ' << y0;
                double step=(y1-y0)/4;
                for(int i=0;i<4;++i)
                {
                    double bulge=face_right?12:-12;
                    path << " C " << x+bulge << ' ' << y0+(i+.15)*step << ' ' << x+bulge << ' ' << y0+(i+.85)*step
                         << ' ' << x << ' ' << y0+(i+1)*step;
                }
                stroke(path.str());
            };
            auto primary_up=b.pins.at(0),primary_low=b.pins.at(2),secondary_up=b.pins.at(1),secondary_low=b.pins.at(3);
            double lx=cx-20,rx=cx+20;
            stroke("M "+std::to_string(primary_up.x)+" "+std::to_string(primary_up.y)+" H "+std::to_string(lx)+" V "+std::to_string(cy-31));
            coil(lx,cy-31,cy+31,true);
            stroke("M "+std::to_string(lx)+" "+std::to_string(cy+31)+" V "+std::to_string(primary_low.y)+" H "+std::to_string(primary_low.x));
            stroke("M "+std::to_string(secondary_up.x)+" "+std::to_string(secondary_up.y)+" H "+std::to_string(rx)+" V "+std::to_string(cy-31));
            coil(rx,cy-31,cy+31,false);
            stroke("M "+std::to_string(rx)+" "+std::to_string(cy+31)+" V "+std::to_string(secondary_low.y)+" H "+std::to_string(secondary_low.x));
            stroke("M "+std::to_string(cx-3)+" "+std::to_string(cy-34)+" V "+std::to_string(cy+34),2.2);
            stroke("M "+std::to_string(cx+3)+" "+std::to_string(cy-34)+" V "+std::to_string(cy+34),2.2);
            if(type=="Tapped Transformer")
            {
                auto tap=b.pins.at(4);
                stroke("M "+std::to_string(rx)+" "+std::to_string(cy)+" H "+std::to_string(tap.x));
            }
        }
        else
        {
            double block_left=b.x+18,block_right=b.x+b.w-18;
            out << "<rect x=\"" << block_left << "\" y=\"" << b.y+7 << "\" width=\"" << block_right-block_left << "\" height=\"" << b.h-14 << "\" fill=\"white\" stroke=\"#111827\" stroke-width=\"1.8\"/>";
            // Put the IEC function designator above the pin-name rows. Keeping
            // it out of the centerline prevents long device names from
            // obscuring the exact terminal labels on compact 4/6/8-pin blocks.
            auto symbol=schematic_symbol(type);
            text(cx-symbol.size()*3.3,b.y+21,symbol,10);
            for (auto const& pin : c.at("pins"))
            {
                int number=pin.at("pin").get<int>();auto p=b.pins.at(number);bool on_left=std::abs(p.x-b.x)<1;
                stroke("M "+std::to_string(p.x)+" "+std::to_string(p.y)+" H "+std::to_string(on_left?block_left:block_right));
                auto name=utf8_ellipsize(schematic_pin_label(c,number),10);
                text(on_left?block_left+4:block_right-4-name.size()*5.3,p.y-3,name,8,"#374151");
            }
        }
        for (auto const& pin : c.at("pins"))
        {
            int number=pin.at("pin").get<int>();auto p=b.pins.at(number);
            out << "<circle data-terminal=\"" << escaped(id) << ".p" << number << "\" cx=\"" << p.x << "\" cy=\"" << p.y << "\" r=\"1.6\" fill=\"#111827\"/>";
        }
        out << "</g>";
    }
    out << "<path d=\"M 1160 108 V 1012\" stroke=\"#cbd5dd\"/>";
    text(1182, 135, "Schematic index", 17);
    text(1182, 163, std::to_string(count) + "/" + std::to_string(global.at("components").size()) + " components selected", 13);
    text(1182, 185, std::to_string(node_terms.size()) + " exact nodes", 13);
    text(1182, 207, std::to_string(junctions) + " shared junction hubs", 13);
    text(1182, 229, std::to_string(unconnected) + " unconnected pins", 13);
    text(1182, 251, std::to_string(external) + " off-view node stubs", 13);
    text(1182, 283, "IEC blocks: uncommon/multipin parts", 11, "#526675");
    text(1182, 302, "Wires: exact node membership", 11, "#526675");
    text(1182, 321, "Crossings: never junctions", 11, "#526675");
    text(1182, 340, "Values: saved/state data", 11, "#526675");
    std::set<std::string> visible;
    for (auto const& [id, ignored] : boxes) visible.insert(id);
    json locator;
    if (visible.size() < global.at("components").size()) locator = render_locator(global, visible, out, 1182, 760, 286, 222);
    text(28, 1018, "Read-only schematic projection | exact IDs/nodes in JSON | functional claims require solver measurements", 12, "#526675");
    out << "</g></svg>";
    json result{{"source", "position-aware-auto-wired-schematic"}, {"image_generated", true},
        {"schematic", true}, {"width", canvas_width}, {"height", canvas_height},
        {"rendered_components", count}, {"total_components", global.at("components").size()},
        {"rendered_nodes", node_terms.size()}, {"routed_nodes", routed},
        {"junction_count", junctions}, {"unconnected_pin_count", unconnected},
        {"external_connection_stub_count", external}, {"geometry_mutated", false},
        {"layout_source", "conventional symbols in non-overlapping rank-preserving cells derived from original native xyz"},
        {"routing", "black orthogonal paths grouped by exact saved/native node; filled hubs only for nodes with >=3 total terminals"},
        {"warnings", {"Schematic spacing is a view-only transformation; use native positions from JSON for metric/spatial claims",
                      "Wire crossings without a filled node hub are not electrically connected"}}};
    if (!locator.is_null()) result["minimap"] = locator;
    return result;
}

json render_spatial(json const& data, json const& global, std::ostream& out, std::size_t offset, std::size_t limit, bool top, json const& camera_options)
{
    auto const& cs = data.at("components");
    auto end = std::min(cs.size(), offset + limit);
    if (offset >= cs.size()) throw std::runtime_error("page offset exceeds component count");
    std::vector<pl::position> points;
    std::map<std::string, std::size_t> selected;
    bool generated = false;
    for (std::size_t i = offset; i < end; ++i)
    {
        selected[cs[i]["id"].get<std::string>()] = i;
        generated |= cs[i]["position_source"] == "generated";
        for (double x : {-.06, .06}) for (double y : {-.04, .04}) for (double z : {-.015, .025}) points.push_back(local_to_world(cs[i], {x, y, z}));
    }
    std::vector<int> legend_y;
    int legend_bottom = 135;
    auto legend_end = offset;
    bool reserve_locator = end-offset < global["components"].size() || !camera_options.empty();
    for (std::size_t i = offset; i < std::min(end, offset + 4); ++i)
    {
        auto card_height = 100 + static_cast<int>((std::min<std::size_t>(8,cs[i]["pins"].size()) + 1) / 2) * 18;
        if(legend_bottom+card_height>(reserve_locator?545:730)) break;
        legend_y.push_back(legend_bottom);
        legend_bottom += card_height;
        ++legend_end;
    }
    // The diagram, not a repeated component list, owns the canvas dimensions.
    // Long pin maps remain authoritative in the JSON and compact tool details.
    int height = 820;
    auto camera = make_camera(data.value("saved_camera", json::object()), camera_options, points, top, height);
    double focal = (height - 290) / (2 * std::tan(camera.fov * std::acos(-1.0) / 360));
    auto screen = [&](pl::position p) -> xy {
        auto q = add(p, mul(camera.position, -1));
        double scale = camera.perspective ? focal / std::max(camera.near_plane, dot(q, camera.forward)) : (height - 290) / camera.ortho_height;
        scale *= camera.zoom;
        return {365 + dot(q, camera.right) * scale, (height + 90) * .5 - dot(q, camera.up) * scale};
    };
    if (camera_options.empty() || (camera_options.value("mode", "saved") == "saved" && !camera_options.contains("fit") && !camera_options.contains("position") && !camera_options.contains("target") && !camera_options.contains("zoom")))
    {
        bool outside = false;
        double lo_x=1e99,hi_x=-1e99,lo_y=1e99,hi_y=-1e99;
        for(auto p:points){auto s=screen(p);outside|=camera.depth(p)<=camera.near_plane||s.x<25||s.x>705||s.y<125||s.y>775;lo_x=std::min(lo_x,s.x);hi_x=std::max(hi_x,s.x);lo_y=std::min(lo_y,s.y);hi_y=std::max(hi_y,s.y);}
        if(outside || (hi_x-lo_x)*(hi_y-lo_y)<530*530*.08)
        {
            auto fitted=camera_options;fitted["fit"]=true;
            camera=make_camera(data.value("saved_camera",json::object()),fitted,points,top,height);
            camera.warnings.push_back("Automatic framing fallback: saved/default view was clipped or mostly empty; only view target/distance changed, original camera and component transforms are preserved");
        }
    }
    auto camera_result = camera.metadata();
    camera_result["requested"] = camera_options;
    camera_result["fov_source"] = camera_options.contains("fov_y_deg") ? "explicit-renderer-option" : "renderer-default-not-saved";
    camera_result["projected_centers"] = json::array();
    camera_result["clipped_components"] = json::array();
    camera_result["behind_camera"] = json::array();
    camera_result["width"] = 1080; camera_result["height"] = height;
    camera_result["legend_components"] = legend_end-offset;
    camera_result["omitted_legend_components"] = end-legend_end;
    std::set<std::string> renderable;
    for (std::size_t i = offset; i < end; ++i)
    {
        auto p = local_to_world(cs[i], {});
        auto s = screen(p);
        auto id = cs[i]["id"].get<std::string>();
        bool behind = camera.depth(p) <= camera.near_plane;
        bool outside = behind || s.x < 12 || s.x > 718 || s.y < 112 || s.y > height - 38;
        camera_result["projected_centers"].push_back({{"id",id}, {"ref",cs[i]["ref"]}, {"x",s.x}, {"y",s.y}, {"depth",camera.depth(p)}, {"outside_frame",outside}});
        if (outside) camera_result["clipped_components"].push_back(id);
        if (behind) camera_result["behind_camera"].push_back(id);
        else if (!outside) renderable.insert(id);
    }
    if (!camera_result["clipped_components"].empty()) camera_result["warnings"].push_back("Some selected component centers are outside this camera frame; use camera.fit=true or change position/target/zoom. The full netlist is unchanged.");
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"" << height << "\" viewBox=\"0 0 1080 " << height << "\">";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#f7f9fa\"/><g font-family=\"DejaVu Sans,sans-serif\" fill=\"#182936\">";
    auto text = [&](double x, double y, std::string const& s, int size = 13, char const* fill = "#182936") {
        out << "<text x=\"" << x << "\" y=\"" << y << "\" fill=\"" << fill << "\" font-size=\"" << size << "\">" << escaped(s) << "</text>";
    };
    text(22, 30, "Aurex | spatial circuit view", 23);
    text(22, 54, "Camera: " + camera.source + " | " + (camera.perspective ? "perspective" : "orthographic") + ". Native component positions/rotations preserved.");
    text(22, 75, generated ? "Some coordinates were generated (no position provided). IDs/pins/values are authoritative." : "Bodies and terminal geometry are schematic. Saved layout is not rearranged.", 12);
    text(22, 95, "Dot = terminal; crossing wires are not junctions. Use topology view for dense connectivity.", 12);
    out << "<defs><clipPath id=\"scene\"><rect x=\"12\" y=\"112\" width=\"706\" height=\"" << height - 150 << "\"/></clipPath></defs><g clip-path=\"url(#scene)\">";
    // Quiet screen grid only: it carries no invented physical measurements.
    for (int x = 25; x < 719; x += 35) out << "<path d=\"M " << x << " 115 V " << height - 35 << "\" stroke=\"#e8edf0\"/>";
    for (int y = 115; y < height - 34; y += 35) out << "<path d=\"M 12 " << y << " H 718\" stroke=\"#e8edf0\"/>";
    std::size_t off_page_wires = 0;
    std::ostringstream wire_layer;
    for (auto const& wire : data["wires"])
    {
        auto a = selected.find(wire["Source"].get<std::string>()), b = selected.find(wire["Target"].get<std::string>());
        if (a == selected.end() || b == selected.end() || !renderable.contains(wire["Source"]) || !renderable.contains(wire["Target"])) { if (a != selected.end() || b != selected.end()) ++off_page_wires; continue; }
        auto p = screen(port_position(cs[a->second], wire["SourcePin"]));
        auto q = screen(port_position(cs[b->second], wire["TargetPin"]));
        double lift = std::min(35.0, std::abs(p.x - q.x) * .13 + 10);
        wire_layer << "<path data-source=\"" << escaped(wire["Source"].get<std::string>()) << "\" data-target=\"" << escaped(wire["Target"].get<std::string>())
            << "\" d=\"M " << p.x << ' ' << p.y << " C " << p.x << ' ' << p.y - lift << ' ' << q.x << ' ' << q.y - lift << ' ' << q.x << ' ' << q.y
            << "\" stroke=\"#8e792d\" stroke-width=\"2.6\" opacity=\".85\" fill=\"none\"/>";
    }
    auto polygon = [&](json const& c, std::vector<pl::position> const& vertices, char const* fill) {
        out << "<polygon points=\"";
        for (auto p : vertices) { auto q = screen(local_to_world(c, p)); out << q.x << ',' << q.y << ' '; }
        out << "\" fill=\"" << fill << "\" stroke=\"#205276\" stroke-width=\"1.2\"/>";
    };
    std::vector<std::size_t> paint;
    for (std::size_t i = offset; i < end; ++i) if (renderable.contains(cs[i]["id"])) paint.push_back(i);
    std::stable_sort(paint.begin(), paint.end(), [&](auto a, auto b) { return camera.depth(local_to_world(cs[a], {})) > camera.depth(local_to_world(cs[b], {})); });
    for (auto i : paint)
    {
        auto const& c = cs[i];
        auto center = screen(local_to_world(c, {}));
        out << "<g data-component-id=\"" << escaped(c["id"].get<std::string>()) << "\" data-position=\"" << escaped(c["position"].dump()) << "\" data-rotation=\"" << escaped(c["rotation"].dump())
            << "\" data-center-x=\"" << center.x << "\" data-center-y=\"" << center.y << "\">";
        struct face { std::vector<pl::position> vertices; char const* fill; double depth{}; };
        std::vector<face> faces{
            {{{-.046, -.025, -.012}, {-.046, .025, -.012}, {-.046, .025, .012}, {-.046, -.025, .012}}, "#286895"},
            {{{.046, -.025, -.012}, {.046, .025, -.012}, {.046, .025, .012}, {.046, -.025, .012}}, "#286895"},
            {{{-.046, -.025, -.012}, {.046, -.025, -.012}, {.046, -.025, .012}, {-.046, -.025, .012}}, "#205982"},
            {{{-.046, .025, -.012}, {.046, .025, -.012}, {.046, .025, .012}, {-.046, .025, .012}}, "#205982"},
            {{{-.046, -.025, -.012}, {.046, -.025, -.012}, {.046, .025, -.012}, {-.046, .025, -.012}}, "#367fac"},
            {{{-.046, -.025, .012}, {.046, -.025, .012}, {.046, .025, .012}, {-.046, .025, .012}}, "#367fac"},
        };
        for (auto& face : faces) for (auto p : face.vertices)
        {
            auto world = local_to_world(c, p);
            face.depth -= camera.depth(world);
        }
        // Render all transformed faces far-to-near. Hardcoding the original
        // +X/+Y sides leaves holes when PL's default 180-degree rotation flips
        // the visible sides; depth ordering preserves a solid body at any pose.
        std::stable_sort(faces.begin(), faces.end(), [](face const& a, face const& b) { return a.depth < b.depth; });
        for (auto const& face : faces) polygon(c, face.vertices, face.fill);
        auto type = c["type"].get<std::string>();
        auto surface_line = [&](std::vector<pl::position> const& points) {
            out << "<polyline points=\"";
            for (auto p : points) { auto q = screen(local_to_world(c, p)); out << q.x << ',' << q.y << ' '; }
            out << "\" fill=\"none\" stroke=\"#eff8ff\" stroke-width=\"1.7\"/>";
        };
        auto symbol_pos = screen(local_to_world(c, {0, 0, .016}));
        if (type == "Resistor") surface_line({{-.028, 0, .016}, {-.021, 0, .016}, {-.016, -.008, .016}, {-.008, .008, .016}, {0, -.008, .016}, {.008, .008, .016}, {.016, -.008, .016}, {.021, 0, .016}, {.028, 0, .016}});
        else if (type == "Basic Capacitor")
        {
            surface_line({{-.026, 0, .016}, {-.006, 0, .016}});
            surface_line({{-.006, -.013, .016}, {-.006, .013, .016}});
            surface_line({{.006, -.013, .016}, {.006, .013, .016}});
            surface_line({{.006, 0, .016}, {.026, 0, .016}});
        }
        else
        {
            std::string symbol;
            if (type == "Battery Source") symbol = "+  -";
            else if (type == "Basic Inductor") symbol = "L";
            else if (type == "Logic Input") symbol = "IN";
            else if (type == "Logic Output") symbol = "OUT";
            else if (type == "Ground Component")
            {
                out << "<path d=\"M " << symbol_pos.x << ' ' << symbol_pos.y - 13 << " V " << symbol_pos.y - 2
                    << " M " << symbol_pos.x - 13 << ' ' << symbol_pos.y - 2 << " H " << symbol_pos.x + 13
                    << " M " << symbol_pos.x - 9 << ' ' << symbol_pos.y + 4 << " H " << symbol_pos.x + 9
                    << " M " << symbol_pos.x - 4 << ' ' << symbol_pos.y + 10 << " H " << symbol_pos.x + 4
                    << "\" fill=\"none\" stroke=\"#eff8ff\" stroke-width=\"2.2\"/>";
            }
            else if (type == "D Flipflop") symbol = "DFF";
            else if (type.ends_with(" Gate"))
            {
                symbol = type.substr(0, type.size() - 5);
                std::transform(symbol.begin(), symbol.end(), symbol.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
                if (symbol == "NO") symbol = "NOT";
            }
            if (!symbol.empty()) text(symbol_pos.x - symbol.size() * 3.7, symbol_pos.y + 3, symbol, 12, "#eff8ff");
        }
        out << "</g>";
    }
    // Wires run above the simplified bodies so both exact terminals stay
    // visibly connected, rather than appearing to terminate at a box edge.
    out << wire_layer.str();
    struct annotation_box { double x, y, w, h; };
    std::vector<annotation_box> pin_labels;
    for (auto i : paint)
    {
        auto const& c = cs[i];
        for (auto const& pin : c["pins"])
        {
            auto p = screen(port_position(c, pin["pin"]));
            out << "<circle cx=\"" << p.x << "\" cy=\"" << p.y << "\" r=\"4.6\" fill=\"#ead59b\" stroke=\"#6d551e\" stroke-width=\"1.2\"/>";
            if(paint.size()>6 || c["pins"].size()>8) continue;
            auto label = c["ref"].get<std::string>() + ".p" + std::to_string(pin["pin"].get<int>()) + ":" + pin["node"].get<std::string>();
            auto center = screen(local_to_world(c, {}));
            double width = label.size() * 8.7 + 6;
            double label_x = p.x < center.x ? p.x - 10 - width : p.x + 9;
            label_x = std::clamp(label_x, 18.0, 706.0 - width);
            double label_y = p.y - 25;
            for (int attempt = 0; attempt < 20; ++attempt)
            {
                bool collides = false;
                for (auto const& box : pin_labels) collides |= label_x < box.x + box.w + 3 && label_x + width + 3 > box.x && label_y < box.y + box.h + 3 && label_y + 25 > box.y;
                if (!collides) break;
                label_y -= 25;
                if (label_y < 116) label_y = p.y + 12 + attempt * 25;
            }
            pin_labels.push_back({label_x, label_y, width, 22});
            out << "<path d=\"M " << p.x << ' ' << p.y << " L " << std::clamp(p.x, label_x, label_x + width) << ' ' << label_y + 22 << "\" stroke=\"#7c8e9c\" stroke-width=\".8\"/>";
            out << "<rect x=\"" << label_x - 3 << "\" y=\"" << label_y << "\" width=\"" << width << "\" height=\"22\" rx=\"3\" fill=\"#ffffff\" fill-opacity=\".96\"/>";
            text(label_x, label_y + 17, label, 16);
        }
    }
    // Short high-contrast identities remain readable after the application
    // resizes the image to 768px; they are annotations, never layout changes.
    for (auto i : paint)
    {
        if(paint.size()>8) continue;
        auto const& c = cs[i];
        auto center = screen(local_to_world(c, {}));
        double top_edge = center.y, bottom_edge = center.y;
        for (double x : {-.046, .046}) for (double y : {-.025, .025}) for (double z : {-.012, .012})
        {
            double edge = screen(local_to_world(c, {x, y, z})).y;
            top_edge = std::min(top_edge, edge);
            bottom_edge = std::max(bottom_edge, edge);
        }
        std::string name = c["label"].get<std::string>();
        // Useful short native IDs may be drawn as a fallback without turning
        // them into a saved Label or a falsely named public port.
        if (name.empty() && c["id"].get<std::string>().size() <= 14) name = c["id"].get<std::string>();
        if (c["type"] == "Ground Component") name = "GND";
        if (name.empty() || name.size() > 14)
        {
            name = c["type"].get<std::string>();
            if (name == "Logic Input") name = "IN";
            else if (name == "Logic Output") name = "OUT";
            else if (name == "Resistor") name = "R";
            else if (name == "Battery Source") name = "V";
            else if (name.size() > 14) name = name.substr(0, 14);
        }
        auto label = c["ref"].get<std::string>() + " " + name;
        if (c["type"] == "Ground Component") label += " (1 pin)";
        if (c["type"] == "Resistor" && c["properties"].contains("电阻"))
        {
            std::ostringstream r; r.precision(5); r << c["properties"]["电阻"].get<double>();
            label += " / " + r.str() + " ohm";
        }
        double width = std::max(75.0, label.size() * 8.3 + 14);
        double x = std::clamp(center.x - width * .5, 18.0, 706.0 - width);
        double y = c["type"] == "Ground Component" ? top_edge - 35 : bottom_edge + 8;
        y = std::clamp(y, 116.0, static_cast<double>(height - 67));
        auto preferred_y=y;
        for(int attempt=0;attempt<24;++attempt)
        {
            bool collides=false;
            for(auto const&box:pin_labels) collides|=x<box.x+box.w+3&&x+width+3>box.x&&y<box.y+box.h+3&&y+30>box.y;
            if(!collides)break;
            y=std::clamp(preferred_y+(attempt%2?-1:1)*((attempt/2)+1)*32.0,116.0,static_cast<double>(height-67));
        }
        out << "<path d=\"M " << center.x << ' ' << (y > center.y ? y : y + 27) << " L " << center.x << ' ' << center.y << "\" stroke=\"#8ea5b4\" stroke-width=\".9\"/>";
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << width << "\" height=\"27\" rx=\"4\" fill=\"white\" stroke=\"#bac9d3\"/>";
        text(x + 7, y + 19, label, 16);
        pin_labels.push_back({x,y,width,27});
    }
    std::size_t label_overlap_count=0;
    for(std::size_t i=0;i<pin_labels.size();++i) for(std::size_t j=i+1;j<pin_labels.size();++j)
    {
        auto const&a=pin_labels[i];auto const&b=pin_labels[j];
        label_overlap_count+=a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
    }
    camera_result["label_overlap_count"]=label_overlap_count;
    if(label_overlap_count) camera_result["warnings"].push_back("Projected pin/identity labels overlap; no clipping does NOT imply legibility. Focus fewer components or adjust camera angle/zoom, and use authoritative pin mappings.");
    out << "</g>";
    out << "<path d=\"M 735 113 V " << height - 22 << "\" stroke=\"#cbd5dd\"/>";
    for (std::size_t i = offset; i < legend_end; ++i)
    {
        auto const& c = cs[i];
        int y = legend_y[i - offset];
        text(751, y, c["ref"].get<std::string>() + "  " + (c["label"].get<std::string>().empty() ? c["type"].get<std::string>() : c["label"].get<std::string>()), 15);
        text(751, y + 19, c["type"].get<std::string>(), 12);
        text(751, y + 37, compact_properties(c), 12);
        if (c["native"].contains("measurements"))
        {
            auto const& m = c["native"]["measurements"];
            std::ostringstream value; value.precision(4);
            if (m.contains("voltage_across_0_to_1"))
            {
                auto const& v = m["voltage_across_0_to_1"];
                value << "V(p0)-V(p1)=" << v["real"].get<double>();
                double imag = v["imag"].get<double>();
                if (imag) value << (imag < 0 ? "-j" : "+j") << std::abs(imag);
                value << " V";
                if (m.contains("derived_current_0_to_1"))
                {
                    auto const& current = m["derived_current_0_to_1"];
                    value << "; I=" << current["real"].get<double>();
                    double current_imag = current["imag"].get<double>();
                    if (current_imag) value << (current_imag < 0 ? "-j" : "+j") << std::abs(current_imag);
                    value << " A";
                }
            }
            else if (c["native"].value("type", "").starts_with("digital_")) { value << "Measured logic "; for (auto const& d : m["digital"]) value << "01XZ"[d.get<unsigned>() % 4] << ' '; }
            text(751, y + 55, value.str(), 11);
        }
        text(751, y + 76, std::to_string(c["pins"].size()) + (c["pins"].size() == 1 ? " pin:" : " pins:"), 13);
        std::string pins;
        std::size_t pin_index = 0;
        for (auto const& pin : c["pins"])
        {
            if(pin_index>=8) break;
            if (!pins.empty()) pins += "; ";
            pins += c["ref"].get<std::string>() + ".p" + std::to_string(pin["pin"].get<int>()) + "=" + pin["node"].get<std::string>();
            if (++pin_index % 2 == 0 || pin_index == c["pins"].size())
            {
                text(751, y + 94 + static_cast<int>((pin_index - 1) / 2) * 18, pins, 13);
                pins.clear();
            }
        }
    }
    if(end>legend_end)
    {
        text(751, reserve_locator?555:745, std::to_string(end-legend_end)+" more selected components", 13);
        text(751, reserve_locator?576:766, "Focus 1-4 IDs for readable pins/values.", 12);
        camera_result["warnings"].push_back("Dense selection: per-pin labels and repeated legend entries are intentionally omitted; use focus_ids with 1-4 components for electrical details");
    }
    text(22, height - 15, (data.value("focused", false) ? "Focused components " : "Components ") + std::to_string(offset + 1) + "-" + std::to_string(end) + "/" + std::to_string(data.value("total_components", cs.size())) + "; " + std::to_string(off_page_wires) + " external connections; " + std::to_string(camera_result["clipped_components"].size()) + " outside camera. Full native xyz/pins in JSON.", 11);
    if(reserve_locator && renderable.size()<global["components"].size())
        camera_result["minimap"]=render_locator(global,renderable,out,751,590,307,200);
    out << "</g></svg>";
    return camera_result;
}

// Publication covers are a server-owned policy, not an agent's inspection
// camera. Fixed canvas and all-component framing keep large digital circuits
// bounded and prevent a previous local focus from becoming the public cover.
std::set<std::string> primary_cluster(json const& components)
{
    std::set<std::string> selected;
    std::array<double,3> lo{},hi{};
    for(std::size_t axis=0;axis<3;++axis)
    {
        std::vector<double> values;
        for(auto const&c:components) values.push_back(c["position"][axis].get<double>());
        std::sort(values.begin(),values.end());
        double q10=values[values.size()/10],q90=values[values.size()*9/10];
        double extent=std::max(.2,q90-q10);
        lo[axis]=q10-3*extent;hi[axis]=q90+3*extent;
    }
    for(auto const&c:components)
    {
        bool inside=true;
        for(std::size_t a=0;a<3;++a) inside&=c["position"][a]>=lo[a]&&c["position"][a]<=hi[a];
        if(inside) selected.insert(c["id"]);
    }
    // This is only a labelled viewport suggestion, never removal from saved
    // data or the mandatory all-components publication cover.
    if(components.size()<32 || selected.size()*5<components.size()*4)
        for(auto const&c:components) selected.insert(c["id"]);
    return selected;
}

json render_overview(json const& original, std::ostream& out, bool publication=true, json const& options=json::object(), bool region=false)
{
    auto data=original;
    if(original["components"].empty()) throw std::runtime_error("cannot render an empty overview");
    auto cluster=primary_cluster(original["components"]);
    json excluded=json::array();
    for(auto const&c:original["components"]) if(!cluster.contains(c["id"])) excluded.push_back({{"id",c["id"]},{"ref",c["ref"]},{"type",c["type"]},{"position",c["position"]}});
    std::size_t external=0;
    if(region && !publication)
    {
        data["components"]=json::array();data["wires"]=json::array();
        for(auto const&c:original["components"]) if(cluster.contains(c["id"])) data["components"].push_back(c);
        for(auto const&w:original["wires"])
        {
            if(cluster.contains(w["Source"])&&cluster.contains(w["Target"])) data["wires"].push_back(w);
            else ++external;
        }
    }
    auto const& cs = data.at("components");
    if (cs.empty()) throw std::runtime_error("cannot render an empty publication cover");
    bool reserve_locator=!publication&&(region||!options.empty());
    double frame_height=reserve_locator?600:780, frame_center_y=reserve_locator?405:510;
    double scene_bottom=reserve_locator?710:920;
    std::vector<pl::position> points;
    for (auto const& c : cs) for (double x : {-.06,.06}) for (double y : {-.04,.04}) for (double z : {-.015,.025}) points.push_back(local_to_world(c,{x,y,z}));
    auto camera = publication
        ? make_camera(json::object(), {{"mode","custom"}, {"yaw_deg",45}, {"pitch_deg",60}, {"projection","orthographic"}, {"fit",true}}, points, false, 960)
        : make_camera(data.value("saved_camera",json::object()), options, points, false, 960,1160,frame_height);
    if(publication) camera.source = "fixed-publication-overview";
    double xmin=1e99,xmax=-1e99,ymin=1e99,ymax=-1e99;
    for (auto p : points)
    {
        double x=dot(p,camera.right), y=-dot(p,camera.up);
        xmin=std::min(xmin,x); xmax=std::max(xmax,x); ymin=std::min(ymin,y); ymax=std::max(ymax,y);
    }
    double cx=(xmin+xmax)*.5,cy=(ymin+ymax)*.5;
    double scale=std::min(1160/std::max(.01,xmax-xmin),780/std::max(.01,ymax-ymin))*.98;
    if(publication)
    {
        camera.target=add(camera.target,add(mul(camera.right,cx-dot(camera.target,camera.right)),mul(camera.up,-cy-dot(camera.target,camera.up))));
        camera.position=add(camera.target,mul(camera.forward,-camera.distance));
        camera.ortho_height=780/scale;
    }
    double pan_x=0,pan_y=0;
    auto screen=[&](pl::position p)->xy {
        if(publication)return {640+(dot(p,camera.right)-cx)*scale,510+(-dot(p,camera.up)-cy)*scale};
        auto q=add(p,mul(camera.position,-1));
        double k=camera.perspective?(frame_height*.5)/std::tan(camera.fov*std::acos(-1.0)/360)/std::max(camera.near_plane,dot(q,camera.forward)):frame_height/camera.ortho_height;
        return {640+pan_x+dot(q,camera.right)*k*camera.zoom,frame_center_y+pan_y-dot(q,camera.up)*k*camera.zoom};
    };
    if(!publication && (options.empty() || (options.value("mode","saved")=="saved" && !options.contains("fit") && !options.contains("target") && !options.contains("position") && !options.contains("zoom"))))
    {
        bool outside=false; double x0=1e99,x1=-1e99,y0=1e99,y1=-1e99;
        for(auto p:points){auto s=screen(p);outside|=camera.depth(p)<=camera.near_plane||s.x<36||s.x>1244||s.y<105||s.y>scene_bottom;x0=std::min(x0,s.x);x1=std::max(x1,s.x);y0=std::min(y0,s.y);y1=std::max(y1,s.y);}
        if(outside || (x1-x0)*(y1-y0)<1160*frame_height*.08)
        {
            auto fitted=options;fitted["fit"]=true;
            camera=make_camera(data.value("saved_camera",json::object()),fitted,points,false,960,1160,frame_height);
            camera.warnings.push_back("Automatic full-scene framing fallback: saved/default view was clipped or mostly empty; original camera and all positions/rotations are unchanged");
        }
    }
    if(!publication && camera.fit)
    {
        // Perspective depth can make a fitted world-space bounding-box center
        // visibly off-center. A documented image-plane pan centers the actual
        // projection without rotating, moving, or flattening the saved scene.
        double x0=1e99,x1=-1e99,y0=1e99,y1=-1e99;
        for(auto p:points){auto s=screen(p);x0=std::min(x0,s.x);x1=std::max(x1,s.x);y0=std::min(y0,s.y);y1=std::max(y1,s.y);}
        pan_x=640-(x0+x1)*.5;pan_y=frame_center_y-(y0+y1)*.5;
    }
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1280\" height=\"960\" viewBox=\"0 0 1280 960\">"
        << "<rect width=\"1280\" height=\"960\" fill=\"#f7f9fa\"/><g font-family=\"DejaVu Sans,sans-serif\" fill=\"#182936\">"
        << "<text x=\"36\" y=\"43\" font-size=\"28\">Aurex | " << (region?"primary layout viewport (NOT the full scene)":"complete electrical experiment") << "</text>"
        << "<text x=\"36\" y=\"73\" font-size=\"16\">" << cs.size() << " components; " << data["wires"].size()
        << " connections. " << (region?std::to_string(cs.size())+"/"+std::to_string(original["components"].size())+" components; "+std::to_string(external)+" off-viewport connections":publication?"Fixed publication camera":"Complete-scene inspection overview") << "; original layout preserved.</text>"
        << "<defs><clipPath id=\"overview-scene\"><rect x=\"24\" y=\"105\" width=\"1232\" height=\""<<scene_bottom-105<<"\"/></clipPath></defs><g clip-path=\"url(#overview-scene)\">";
    std::vector<std::size_t> paint(cs.size()); std::iota(paint.begin(),paint.end(),0);
    std::stable_sort(paint.begin(),paint.end(),[&](auto a,auto b){return camera.depth(local_to_world(cs[a],{}))>camera.depth(local_to_world(cs[b],{}));});
    std::map<std::string,std::size_t> ids;
    json clipped=json::array();
    for (auto i : paint)
    {
        auto const& c=cs[i]; ids[c["id"].get<std::string>()]=i;
        bool outside=false;
        for (double x : {-.06,.06}) for (double y : {-.04,.04}) for (double z : {-.015,.025})
        { auto p=screen(local_to_world(c,{x,y,z})); outside|=p.x<36||p.x>1244||p.y<105||p.y>scene_bottom; }
        if(outside) clipped.push_back(c["id"]);
        struct face {std::vector<pl::position> vertices; char const* color; double depth{};};
        std::vector<face> faces{
            {{{-.046,-.025,-.012},{-.046,.025,-.012},{-.046,.025,.012},{-.046,-.025,.012}},"#286895"},
            {{{.046,-.025,-.012},{.046,.025,-.012},{.046,.025,.012},{.046,-.025,.012}},"#286895"},
            {{{-.046,-.025,-.012},{.046,-.025,-.012},{.046,-.025,.012},{-.046,-.025,.012}},"#205982"},
            {{{-.046,.025,-.012},{.046,.025,-.012},{.046,.025,.012},{-.046,.025,.012}},"#205982"},
            {{{-.046,-.025,-.012},{.046,-.025,-.012},{.046,.025,-.012},{-.046,.025,-.012}},"#367fac"},
            {{{-.046,-.025,.012},{.046,-.025,.012},{.046,.025,.012},{-.046,.025,.012}},"#367fac"}};
        for(auto& face:faces) for(auto p:face.vertices) face.depth-=camera.depth(local_to_world(c,p));
        std::stable_sort(faces.begin(),faces.end(),[](auto const&a,auto const&b){return a.depth<b.depth;});
        out << "<g data-component-id=\"" << escaped(c["id"].get<std::string>()) << "\" data-position=\"" << escaped(c["position"].dump()) << "\" data-rotation=\"" << escaped(c["rotation"].dump()) << "\">";
        for(auto const& face:faces)
        {
            out << "<polygon points=\"";
            for(auto p:face.vertices){auto q=screen(local_to_world(c,p));out<<q.x<<','<<q.y<<' ';}
            out << "\" fill=\""<<face.color<<"\" stroke=\"#205276\" stroke-width=\".55\"/>";
        }
        out << "</g>";
    }
    for(auto const& wire:data["wires"])
    {
        auto const&a=cs[ids.at(wire["Source"].get<std::string>())],&b=cs[ids.at(wire["Target"].get<std::string>())];
        auto p=screen(port_position(a,wire["SourcePin"])),q=screen(port_position(b,wire["TargetPin"]));
        out<<"<path d=\"M "<<p.x<<' '<<p.y<<" L "<<q.x<<' '<<q.y<<"\" fill=\"none\" stroke=\"#887334\" opacity=\".55\" stroke-width=\".85\"/>";
    }
    if(cs.size()<=256) for(auto const&c:cs) for(auto const&pin:c["pins"])
    {auto p=screen(port_position(c,pin["pin"]));out<<"<circle cx=\""<<p.x<<"\" cy=\""<<p.y<<"\" r=\"1.9\" fill=\"#ead59b\"/>";}
    out << "</g>";
    json locator;
    if(reserve_locator && (cs.size()<original["components"].size()||!clipped.empty()))
    {
        std::set<std::string> visible;
        for(auto const& c:cs) visible.insert(c["id"]);
        for(auto const& id:clipped) visible.erase(id.get<std::string>());
        locator=render_locator(original,visible,out,948,730,307,200);
        out<<"<text x=\"36\" y=\"760\" font-size=\"16\">Local viewport: "<<visible.size()<<'/'<<original["components"].size()<<" components in frame.</text>"
           <<"<text x=\"36\" y=\"790\" font-size=\"14\">Global locator includes distant elements; yellow marks this view, not a changed circuit.</text>"
           <<"<text x=\"36\" y=\"817\" font-size=\"14\">This rough locator is for orientation only. Use IDs/netlist for exact connections.</text>";
    }
    out << "<text x=\"36\" y=\"942\" font-size=\"13\">Overview is not pin-level evidence. Physical occlusion is possible; focus IDs to inspect connections and values.</text></g></svg>";
    auto result=camera.metadata();
    result["frame_pan_pixels"]={pan_x,pan_y};
    result["overview"]=true; result["width"]=1280; result["height"]=960;
    result["total_components"]=original["components"].size(); result["rendered_components"]=cs.size();
    result["primary_cluster_component_count"]=cluster.size();result["spatial_outliers"]=excluded;
    result["viewport_is_subset"]=region;result["external_connections"]=external;
    if(region)result["warnings"].push_back("Primary layout viewport excludes explicitly listed distant components, NOT deleted or moved. Consult the separate full-scene overview and original netlist for complete layout/connectivity.");
    result["clipped_component_ids"]=clipped; result["clipped_components"]=clipped;
    result["occlusion_possible"]=true; result["saved_raw"]=data.value("saved_camera",json::object());
    result["policy"]=publication?"fixed orthographic yaw=45 pitch=60; fit every component; ignore agent camera/focus/pagination":"full-scene overview, saved/default camera with explicit fit fallback if unusable; no component pagination or rearrangement";
    result["label_overlap_count"]=0;
    if(!locator.is_null()) result["minimap"]=locator;
    result["labels_omitted_for_overview"]=true;
    if(!publication) result["warnings"].push_back("Full-scene overview omits individual pin/value labels; no label overlap does not imply pin-level legibility. Use focus_ids or query for a local detail view");
    return result;
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 5) throw std::runtime_error("usage: circuit_view render|create|state INPUT OUTPUT.svg OUTPUT.netlist.json [OFFSET] [LIMIT] [OUTPUT.sav|-] [auto|overview|region|spatial|topology|schematic] [isometric|top] [FOCUS_IDS_JSON] [QUERY] [CAMERA_JSON] [{\"with_image\":true|false}]");
        auto input = read_json(argv[2]);
        bool with_image = true;
        if (argc > 13)
        {
            auto render_options = json::parse(argv[13]);
            if (!render_options.is_object() || render_options.size() != 1 || !render_options.contains("with_image") || !render_options["with_image"].is_boolean())
                throw std::runtime_error("render options must be {with_image: boolean}");
            with_image = render_options["with_image"].get<bool>();
        }
        std::string mode = argv[1];
        if (mode != "create" && mode != "render" && mode != "state") throw std::runtime_error("mode must be create, render or state");
        json camera_warnings = json::array();
        auto saved_camera = camera_save(input, mode == "state", camera_warnings);
        json camera_options = mode != "render" ? input.value("camera", json::object()) : json::object();
        if (argc > 12)
        {
            auto explicit_options = json::parse(argv[12]);
            if (!explicit_options.is_object() || !camera_options.is_object()) throw std::runtime_error("camera must be an object");
            camera_options.update(explicit_options);
        }
        if (camera_options.contains("overview") && !camera_options["overview"].is_boolean()) throw std::runtime_error("overview must be boolean");
        bool overview=camera_options.value("overview",false);
        // Community responses wrap the original Experiment without repeating
        // its Type at the root. Preserve the file and copy the actual nested
        // type into this in-memory wrapper, never assume electrical Type=0.
        if (input.contains("Experiment") && !input.contains("Type")) input["Type"] = input.at("Experiment").at("Type");
        std::optional<pl::experiment> ex;
        json data;
        if (mode == "state") data = inspect_state(input);
        else
        {
            // The renderer consumes the exact raw camera itself. Removing it
            // only from this in-memory loader copy prevents malformed camera
            // text from blocking a valid electrical netlist. No input is saved.
            auto loader_input = input;
            if (mode == "render")
            {
                if (loader_input.contains("Experiment")) loader_input["Experiment"]["CameraSave"] = "";
                else loader_input["CameraSave"] = "";
            }
            ex = mode == "create" ? from_spec(loader_input) : pl::experiment::load_from_json(loader_input);
            data = inspect(*ex);
        }
        data["saved_camera"] = saved_camera;
        for (auto const& warning : camera_warnings) data["warnings"].push_back(warning);
        std::size_t offset = argc > 5 ? std::stoul(argv[5]) : 0;
        std::size_t limit = argc > 6 ? std::stoul(argv[6]) : 12;
        std::string view = argc > 8 ? argv[8] : "spatial";
        if(view=="auto") view=data["components"].size()>8?"overview":"spatial";
        if(view!="overview" && view!="region" && view!="spatial" && view!="topology" && view!="schematic") throw std::runtime_error("view must be auto, overview, region, spatial, topology or schematic");
        bool complete_scene=overview || view=="overview" || view=="region";
        if (!complete_scene && (!limit || limit > 24)) throw std::runtime_error("page limit must be 1..24");
        if(complete_scene){offset=0;limit=data["components"].size();}
        std::ofstream svg;
        if (with_image) { svg.open(argv[3]); if (!svg) throw std::runtime_error("cannot write SVG"); }
        std::string projection = argc > 9 ? argv[9] : "isometric";
        auto visual_data = data;
        std::set<std::string> targets;
        json focus = argc > 10 ? json::parse(argv[10]) : json::array();
        if(complete_scene) focus=json::array();
        if (!focus.is_array() || focus.size() > limit) throw std::runtime_error("focus_ids must be an array no larger than page limit");
        std::string query = argc > 11 ? argv[11] : "";
        if(complete_scene) query.clear();
        auto lower = [](std::string s) { for (auto& ch : s) if (static_cast<unsigned char>(ch) < 128) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); return s; };
        query = lower(query);
        // Resolve each exact focus token before pagination and text search.
        // A display Label must never shadow an actual Identifier or ref.
        std::set<std::string> resolved_focus;
        std::vector<std::string> primary_order;
        for (auto const& token : focus)
        {
            if (!token.is_string() || token.get<std::string>().empty()) throw std::runtime_error("focus_ids entries must be nonempty strings");
            bool found = false;
            for (auto key : {"id", "ref", "label"})
            {
                std::vector<json const*> matches;
                for (auto const& c : data["components"]) if (c[key] == token) matches.push_back(&c);
                if (matches.empty()) continue;
                if (matches.size() != 1)
                {
                    std::string message = "Ambiguous focus " + std::string(key) + " " + token.dump().substr(0, 140) + "; use an exact unique Identifier. Candidates: ";
                    for (std::size_t i = 0; i < std::min<std::size_t>(8, matches.size()); ++i)
                    {
                        if (i) message += ", ";
                        message += matches[i]->at("ref").get<std::string>() + "=" + matches[i]->at("id").dump();
                    }
                    if (matches.size() > 8) message += " (more matches omitted)";
                    throw std::runtime_error(message);
                }
                auto id = matches[0]->at("id").get<std::string>();
                if (resolved_focus.insert(id).second) primary_order.push_back(id);
                found = true;
                break;
            }
            if (!found) throw std::runtime_error("Unknown focus token " + token.dump().substr(0,140) + "; use an exact Identifier, ref or unique Label");
        }
        // Pagination applies to one deterministic, deduplicated union: exact
        // focus in request order first, then query matches in original order.
        for (auto const& c : data["components"])
        {
            bool match = !query.empty() && lower(c["id"].get<std::string>() + " " + c["label"].get<std::string>() + " " + c["type"].get<std::string>() + " " + c["properties"].dump()).find(query) != std::string::npos;
            if (match && !resolved_focus.contains(c["id"].get<std::string>())) primary_order.push_back(c["id"].get<std::string>());
        }
        std::size_t match_count = primary_order.size();
        bool focused = !focus.empty() || !query.empty();
        json selection = json::object();
        if (focused)
        {
            if (primary_order.empty()) throw std::runtime_error("No components match focus_ids/query");
            if (offset >= primary_order.size()) throw std::runtime_error("focus/query offset exceeds total_matches=" + std::to_string(primary_order.size()));
            auto page_end = std::min(primary_order.size(), offset + limit);
            json primary_ids = json::array(), primary_refs = json::array(), neighbor_ids = json::array(), neighbor_refs = json::array();
            std::map<std::string, std::string> refs;
            for (auto const& c : data["components"]) refs.emplace(c["id"].get<std::string>(), c["ref"].get<std::string>());
            for (std::size_t i = offset; i < page_end; ++i)
            {
                targets.insert(primary_order[i]);
                primary_ids.push_back(primary_order[i]); primary_refs.push_back(refs.at(primary_order[i]));
            }
            auto selected = targets;
            for (auto const& w : data["wires"])
            {
                if (selected.size() == limit) break;
                if (targets.contains(w["Source"])) selected.insert(w["Target"]);
                if (selected.size() == limit) break;
                if (targets.contains(w["Target"])) selected.insert(w["Source"]);
            }
            visual_data["components"] = json::array();
            visual_data["focused"] = true;
            visual_data["total_components"] = data["components"].size();
            for (auto const& c : data["components"]) if (selected.contains(c["id"])) visual_data["components"].push_back(c);
            for (auto const& id : selected) if (!targets.contains(id)) { neighbor_ids.push_back(id); neighbor_refs.push_back(refs.at(id)); }
            selection = {{"total_matches",match_count},{"offset",offset},{"limit",limit},{"primary_ids",primary_ids},{"primary_refs",primary_refs},
                {"neighbor_ids",neighbor_ids},{"neighbor_refs",neighbor_refs},{"has_more",page_end<primary_order.size()},
                {"next_offset",page_end<primary_order.size()?json(page_end):json()},
                {"order","explicit focus in request order, then query matches in saved order; IDs deduplicated before paging"},
                {"scope","Only primary_ids belong to this match page. Optional neighbors fill spare diagram slots and may recur across pages; they are not additional matches."}};
            data["selection"] = selection;
        }
        std::size_t render_offset = focused ? 0 : offset;
        if (projection != "isometric" && projection != "top") throw std::runtime_error("projection must be isometric or top");
        if (!with_image)
            data["camera"] = {{"source","not-rendered"},{"image_generated",false},{"overview",complete_scene},{"saved_raw",saved_camera},{"requested",camera_options},
                {"warnings",{"with_image=false: no SVG/PNG or camera projection was generated. Component positions and the saved camera remain unchanged."}}};
        else if(overview)
        {
            data["camera"]=render_overview(data,svg);
            view="spatial"; projection="fixed-overview";
        }
        else if(view=="overview") data["camera"]=render_overview(data,svg,false,camera_options);
        else if(view=="region") data["camera"]=render_overview(data,svg,false,camera_options,true);
        else if (view == "topology")
        {
            render_topology(visual_data, svg, render_offset, limit);
            data["camera"] = {{"source", "not-applied-topology"}, {"saved_raw", saved_camera}, {"warnings", {"Topology is a connectivity diagram; camera options are not applied"}}};
        }
        else if (view == "schematic")
        {
            data["camera"] = render_schematic(visual_data, data, svg, render_offset, limit);
            data["camera"]["saved_raw"] = saved_camera;
            data["camera"]["requested_camera_ignored"] = camera_options;
            if (!camera_options.empty()) data["camera"]["warnings"].push_back("Camera options are not applied to the position-aware schematic projection");
        }
        else if (view == "spatial") data["camera"] = render_spatial(visual_data, data, svg, render_offset, limit, projection == "top", camera_options);
        else throw std::runtime_error("view must be spatial, overview, topology or schematic");
        std::ofstream netlist(argv[4]);
        if (!netlist) throw std::runtime_error("cannot write netlist");
        netlist << data.dump(2) << '\n';
        if (argc > 7 && std::string(argv[7]) != "-")
        {
            if (mode == "state") throw std::runtime_error("Native snapshot rendering does not export or round-trip a PhysicsLab document");
            std::ofstream sav(argv[7]); if (!sav) throw std::runtime_error("cannot write sav");
            // A source PL document remains byte-semantically unchanged in
            // camera/layout data. Camera adjustment is render-only, never a save edit.
            if (mode == "render") sav << input.dump();
            else
            {
                auto exported=json::parse(ex->dump());
                if(!saved_camera.empty()) exported["Experiment"]["CameraSave"]=saved_camera.dump();
                sav << exported.dump();
            }
        }
        json visible = json::array();
        for (std::size_t i = render_offset; i < std::min(visual_data["components"].size(), render_offset + limit); ++i) visible.push_back(visual_data["components"][i]["id"]);
        std::cout << json{{"components", data["components"].size()}, {"nodes", data["nodes"].size()}, {"offset", offset}, {"limit", limit}, {"view", view}, {"projection", projection},
            {"visible_ids", visible}, {"focused", focused}, {"match_count", match_count}, {"selection",selection},{"with_image",with_image},
            {"camera", data["camera"]}, {"state_origin", data.value("state_origin", json())}}.dump() << '\n';
        return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
