#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/core/type/float16.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>

namespace fs = std::filesystem;

namespace {

class StageError : public std::runtime_error {
public:
    StageError(std::string stage, std::string message)
        : std::runtime_error(std::move(message)),
          m_stage(std::move(stage)) {}

    const std::string& stage() const {
        return m_stage;
    }

private:
    std::string m_stage;
};

[[noreturn]] void fail(const std::string& stage, const std::string& message) {
    throw StageError(stage, message);
}

std::string join(const std::vector<std::string>& values, const std::string& separator) {
    std::ostringstream stream;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << separator;
        }
        stream << values[index];
    }
    return stream.str();
}

template <typename Value>
std::string join_numbers(const std::vector<Value>& values, const std::string& separator = ",") {
    std::ostringstream stream;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << separator;
        }
        stream << values[index];
    }
    return stream.str();
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

size_t checked_product(const ov::Shape& shape, const std::string& context) {
    size_t result = 1;
    for (const size_t value : shape) {
        if (value != 0 && result > std::numeric_limits<size_t>::max() / value) {
            fail("fixture/model validation", "Shape overflow while calculating " + context);
        }
        result *= value;
    }
    return result;
}

std::string shape_string(const ov::Shape& shape) {
    return "[" + join_numbers(std::vector<size_t>(shape.begin(), shape.end())) + "]";
}

std::string partial_shape_string(const ov::PartialShape& shape) {
    std::ostringstream stream;
    stream << shape;
    return stream.str();
}

struct Sha256 {
    static constexpr std::array<uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U,
    };

    void update(const uint8_t* data, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            m_block[m_block_size++] = data[index];
            if (m_block_size == m_block.size()) {
                transform();
                m_bit_count += 512;
                m_block_size = 0;
            }
        }
    }

    std::string finish() {
        uint64_t bits = m_bit_count + static_cast<uint64_t>(m_block_size) * 8;
        m_block[m_block_size++] = 0x80;
        if (m_block_size > 56) {
            while (m_block_size < 64) {
                m_block[m_block_size++] = 0;
            }
            transform();
            m_block_size = 0;
        }
        while (m_block_size < 56) {
            m_block[m_block_size++] = 0;
        }
        for (int offset = 7; offset >= 0; --offset) {
            m_block[m_block_size++] = static_cast<uint8_t>((bits >> (offset * 8)) & 0xffU);
        }
        transform();

        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (const uint32_t value : m_state) {
            stream << std::setw(8) << value;
        }
        return stream.str();
    }

private:
    static uint32_t rotate_right(uint32_t value, uint32_t shift) {
        return (value >> shift) | (value << (32 - shift));
    }

    void transform() {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t offset = index * 4;
            words[index] = (static_cast<uint32_t>(m_block[offset]) << 24) |
                           (static_cast<uint32_t>(m_block[offset + 1]) << 16) |
                           (static_cast<uint32_t>(m_block[offset + 2]) << 8) |
                           static_cast<uint32_t>(m_block[offset + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 =
                rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 =
                rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        uint32_t a = m_state[0];
        uint32_t b = m_state[1];
        uint32_t c = m_state[2];
        uint32_t d = m_state[3];
        uint32_t e = m_state[4];
        uint32_t f = m_state[5];
        uint32_t g = m_state[6];
        uint32_t h = m_state[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
            const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        m_state[0] += a;
        m_state[1] += b;
        m_state[2] += c;
        m_state[3] += d;
        m_state[4] += e;
        m_state[5] += f;
        m_state[6] += g;
        m_state[7] += h;
    }

    std::array<uint8_t, 64> m_block{};
    size_t m_block_size = 0;
    uint64_t m_bit_count = 0;
    std::array<uint32_t, 8> m_state = {
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
};

std::string sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("fixture/model validation", "Cannot open file for SHA-256: " + path.string());
    }
    Sha256 digest;
    std::array<uint8_t, 1024 * 1024> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize size = input.gcount();
        if (size > 0) {
            digest.update(buffer.data(), static_cast<size_t>(size));
        }
    }
    return digest.finish();
}

enum class DataType {
    f16,
    f32,
    i32,
    i64,
};

size_t bytes_per_element(DataType type) {
    switch (type) {
    case DataType::f16:
        return 2;
    case DataType::f32:
    case DataType::i32:
        return 4;
    case DataType::i64:
        return 8;
    }
    throw std::logic_error("Unexpected fixture type");
}

std::string type_string(DataType type) {
    switch (type) {
    case DataType::f16:
        return "f16";
    case DataType::f32:
        return "f32";
    case DataType::i32:
        return "i32";
    case DataType::i64:
        return "i64";
    }
    throw std::logic_error("Unexpected fixture type");
}

DataType parse_data_type(const std::string& value) {
    if (value == "f16") {
        return DataType::f16;
    }
    if (value == "f32") {
        return DataType::f32;
    }
    if (value == "i32") {
        return DataType::i32;
    }
    if (value == "i64") {
        return DataType::i64;
    }
    fail("fixture/model validation", "Unsupported fixture data type: " + value);
}

ov::Shape parse_shape(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    ov::Shape shape;
    for (const std::string& item : split(value, ',')) {
        try {
            const auto parsed = std::stoull(item);
            shape.push_back(static_cast<size_t>(parsed));
        } catch (const std::exception&) {
            fail("fixture/model validation", "Invalid tensor shape: " + value);
        }
    }
    return shape;
}

struct RawTensor {
    std::string name;
    DataType type;
    ov::Shape shape;
    std::vector<uint8_t> bytes;
};

struct Fixture {
    std::map<std::string, std::string> metadata;
    std::map<std::string, RawTensor> tensors;

    const RawTensor& tensor(const std::string& name) const {
        const auto iterator = tensors.find(name);
        if (iterator == tensors.end()) {
            fail("fixture/model validation", "Fixture has no tensor named '" + name + "'");
        }
        return iterator->second;
    }

    size_t size(const std::string& key) const {
        const auto iterator = metadata.find(key);
        if (iterator == metadata.end()) {
            fail("fixture/model validation", "Fixture has no metadata key '" + key + "'");
        }
        try {
            return static_cast<size_t>(std::stoull(iterator->second));
        } catch (const std::exception&) {
            fail("fixture/model validation", "Fixture metadata '" + key + "' is not an integer");
        }
    }
};

Fixture load_fixture(const fs::path& root) {
    const fs::path manifest_path = root / "manifest.txt";
    std::ifstream manifest(manifest_path);
    if (!manifest) {
        fail("fixture/model validation", "Cannot open fixture manifest: " + manifest_path.string());
    }

    Fixture fixture;
    std::string line;
    size_t line_number = 0;
    while (std::getline(manifest, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const size_t separator = line.find('=');
        if (separator == std::string::npos) {
            fail("fixture/model validation",
                 "Invalid manifest line " + std::to_string(line_number) + ": missing '='");
        }
        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key != "tensor") {
            fixture.metadata[key] = value;
            continue;
        }

        const auto pieces = split(value, '|');
        if (pieces.size() != 4) {
            fail("fixture/model validation",
                 "Invalid tensor declaration at manifest line " + std::to_string(line_number));
        }
        RawTensor tensor;
        tensor.name = pieces[0];
        tensor.type = parse_data_type(pieces[1]);
        tensor.shape = parse_shape(pieces[2]);
        const fs::path data_path = root / pieces[3];
        std::ifstream file(data_path, std::ios::binary);
        if (!file) {
            fail("fixture/model validation", "Cannot open fixture tensor: " + data_path.string());
        }
        const size_t expected_size = checked_product(tensor.shape, tensor.name) * bytes_per_element(tensor.type);
        tensor.bytes.resize(expected_size);
        file.read(reinterpret_cast<char*>(tensor.bytes.data()), static_cast<std::streamsize>(tensor.bytes.size()));
        if (static_cast<size_t>(file.gcount()) != expected_size || file.peek() != std::ifstream::traits_type::eof()) {
            fail("fixture/model validation",
                 "Fixture tensor has an unexpected byte length: " + data_path.string());
        }
        if (!fixture.tensors.emplace(tensor.name, std::move(tensor)).second) {
            fail("fixture/model validation", "Duplicate fixture tensor name: " + pieces[0]);
        }
    }
    const auto version = fixture.metadata.find("version");
    if (version == fixture.metadata.end() || version->second != "1") {
        fail("fixture/model validation", "Unsupported or absent fixture manifest version");
    }
    return fixture;
}

float float16_to_float(const uint8_t* data) {
    ov::float16 value;
    std::memcpy(&value, data, sizeof(value));
    return static_cast<float>(value);
}

std::vector<float> raw_to_floats(const RawTensor& tensor) {
    const size_t count = checked_product(tensor.shape, tensor.name);
    std::vector<float> values(count);
    if (tensor.type == DataType::f16) {
        for (size_t index = 0; index < count; ++index) {
            values[index] = float16_to_float(tensor.bytes.data() + index * sizeof(ov::float16));
        }
        return values;
    }
    if (tensor.type == DataType::f32) {
        std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
        return values;
    }
    fail("fixture/model validation", "Tensor '" + tensor.name + "' is not floating point");
}

std::vector<float> tensor_to_floats(const ov::Tensor& tensor) {
    const size_t count = tensor.get_size();
    std::vector<float> values(count);
    if (tensor.get_element_type() == ov::element::f16) {
        const auto* source = tensor.data<const ov::float16>();
        for (size_t index = 0; index < count; ++index) {
            values[index] = static_cast<float>(source[index]);
        }
        return values;
    }
    if (tensor.get_element_type() == ov::element::f32) {
        const auto* source = tensor.data<const float>();
        std::copy(source, source + count, values.begin());
        return values;
    }
    fail("inference", "Expected a floating-point tensor but received " + tensor.get_element_type().to_string());
}

bool compatible_shape(const ov::PartialShape& partial_shape, const ov::Shape& candidate) {
    if (partial_shape.rank().is_dynamic()) {
        return true;
    }
    if (static_cast<size_t>(partial_shape.rank().get_length()) != candidate.size()) {
        return false;
    }
    for (size_t index = 0; index < candidate.size(); ++index) {
        if (partial_shape[index].is_static() &&
            static_cast<size_t>(partial_shape[index].get_length()) != candidate[index]) {
            return false;
        }
    }
    return true;
}

ov::Shape select_input_shape(const RawTensor& source,
                             const std::string& model_name,
                             const ov::PartialShape& target_shape) {
    std::vector<ov::Shape> candidates = {source.shape};
    const auto add = [&](ov::Shape shape) {
        if (std::find(candidates.begin(), candidates.end(), shape) == candidates.end()) {
            candidates.push_back(std::move(shape));
        }
    };
    if (source.shape.size() >= 2 && source.shape[0] == 1) {
        add(ov::Shape(source.shape.begin() + 1, source.shape.end()));
    }
    if (source.shape.size() == 3 && source.shape[0] == 1) {
        if (model_name == "inputs_embeds") {
            add({source.shape[1], source.shape[2]});
        } else if (model_name == "hidden_states") {
            add({source.shape[1], 1, source.shape[2]});
            add({source.shape[1], source.shape[2]});
        }
    }
    if (source.shape.size() == 2 && source.shape[0] == 1) {
        if (model_name == "position_ids" || model_name == "attention_mask") {
            add({source.shape[1]});
        } else if (model_name == "token_type_ids") {
            add({source.shape[1]});
            add({source.shape[1], 1});
        }
    }
    for (const auto& candidate : candidates) {
        if (compatible_shape(target_shape, candidate) &&
            checked_product(candidate, source.name) == checked_product(source.shape, source.name)) {
            return candidate;
        }
    }
    std::vector<std::string> formatted;
    formatted.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        formatted.push_back(shape_string(candidate));
    }
    fail("fixture/model validation",
         "No compatible transformed-port shape for '" + model_name + "': port=" +
             partial_shape_string(target_shape) + ", candidates=" + join(formatted, ", "));
}

ov::Tensor materialize_input(const RawTensor& source,
                             const std::string& model_name,
                             const ov::element::Type& target_type,
                             const ov::PartialShape& target_partial_shape) {
    const ov::Shape selected_shape = select_input_shape(source, model_name, target_partial_shape);
    ov::Tensor target(target_type, selected_shape);
    const size_t count = checked_product(source.shape, source.name);
    if (source.type == DataType::f16 && target_type == ov::element::f32) {
        const auto values = raw_to_floats(source);
        std::copy(values.begin(), values.end(), target.data<float>());
        return target;
    }
    if (source.type == DataType::f16 && target_type == ov::element::f16) {
        std::memcpy(target.data(), source.bytes.data(), source.bytes.size());
        return target;
    }
    if (source.type == DataType::f32 && target_type == ov::element::f32) {
        std::memcpy(target.data(), source.bytes.data(), source.bytes.size());
        return target;
    }
    if (source.type == DataType::i32 && target_type == ov::element::i32) {
        std::memcpy(target.data(), source.bytes.data(), source.bytes.size());
        return target;
    }
    if (source.type == DataType::i64 && target_type == ov::element::i64) {
        std::memcpy(target.data(), source.bytes.data(), source.bytes.size());
        return target;
    }
    fail("fixture/model validation",
         "Cannot bind fixture tensor '" + source.name + "' (" + type_string(source.type) + ") to model port " +
             target_type.to_string() + " with " + std::to_string(count) + " elements");
}

struct Comparison {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;
    size_t failures = 0;
};

Comparison compare(const std::vector<float>& actual,
                   const std::vector<float>& expected,
                   double atol,
                   double rtol) {
    if (actual.size() != expected.size()) {
        fail("numerical comparison",
             "Element count mismatch: actual=" + std::to_string(actual.size()) +
                 ", expected=" + std::to_string(expected.size()));
    }
    Comparison result;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double difference = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double relative = difference / std::max(std::abs(static_cast<double>(expected[index])), 1e-8);
        result.max_abs = std::max(result.max_abs, difference);
        result.max_rel = std::max(result.max_rel, relative);
        result.mean_abs += difference;
        if (difference > atol + rtol * std::abs(static_cast<double>(expected[index]))) {
            ++result.failures;
        }
    }
    if (!actual.empty()) {
        result.mean_abs /= static_cast<double>(actual.size());
    }
    return result;
}

void print_comparison(const std::string& name, const Comparison& comparison, size_t count) {
    std::cout << "  " << name << ": elements=" << count << " max_abs=" << comparison.max_abs
              << " mean_abs=" << comparison.mean_abs << " max_rel=" << comparison.max_rel
              << " failures=" << comparison.failures << '\n';
}

struct Options {
    fs::path model;
    fs::path fixture;
    std::string device = "GPU";
    double atol = 0.1;
    double rtol = 0.1;
    bool dump_pa_ir = false;
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --model <OpenVINO model dir|xml> --fixture <fixture dir> [options]\n"
           "\n"
           "GPU-only options:\n"
           "  --device <GPU[.N]>       GPU device selector (default: GPU)\n"
           "  --atol <value>           Absolute tolerance (default: 0.1)\n"
           "  --rtol <value>           Relative tolerance (default: 0.1)\n"
           "  --dump-pa-ir             Print transformed PagedAttention port contracts\n"
           "  --help                    Show this message\n";
}

double parse_double(const std::string& value, const std::string& option) {
    try {
        const double parsed = std::stod(value);
        if (!std::isfinite(parsed) || parsed < 0.0) {
            throw std::invalid_argument("negative/non-finite");
        }
        return parsed;
    } catch (const std::exception&) {
        fail("arguments", option + " requires a non-negative finite number");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto require_value = [&]() -> std::string {
            if (index + 1 >= argc) {
                fail("arguments", argument + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--model") {
            options.model = require_value();
        } else if (argument == "--fixture") {
            options.fixture = require_value();
        } else if (argument == "--device") {
            options.device = require_value();
        } else if (argument == "--atol") {
            options.atol = parse_double(require_value(), argument);
        } else if (argument == "--rtol") {
            options.rtol = parse_double(require_value(), argument);
        } else if (argument == "--dump-pa-ir") {
            options.dump_pa_ir = true;
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("arguments", "Unknown option: " + argument);
        }
    }
    if (options.model.empty() || options.fixture.empty()) {
        fail("arguments", "--model and --fixture are required");
    }
    if (options.device.rfind("GPU", 0) != 0) {
        fail("arguments", "This simulator is GPU-only; --device must begin with 'GPU'");
    }
    return options;
}

fs::path resolve_model_xml(const fs::path& model) {
    if (fs::is_directory(model)) {
        const fs::path xml = model / "openvino_model.xml";
        if (!fs::is_regular_file(xml)) {
            fail("fixture/model validation", "No openvino_model.xml under " + model.string());
        }
        return xml;
    }
    if (!fs::is_regular_file(model) || model.extension() != ".xml") {
        fail("fixture/model validation", "--model must be an OpenVINO model directory or .xml file");
    }
    return model;
}

std::string port_name(const ov::Output<const ov::Node>& port) {
    try {
        return port.get_any_name();
    } catch (const ov::Exception&) {
        return port.get_node()->get_friendly_name();
    }
}

using InputMap = std::map<std::string, ov::Output<const ov::Node>>;

InputMap inputs_by_name(const ov::CompiledModel& compiled_model) {
    InputMap result;
    for (const auto& input : compiled_model.inputs()) {
        for (const auto& name : input.get_names()) {
            result.emplace(name, input);
        }
    }
    return result;
}

bool is_cache_name(const std::string& name) {
    return name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0;
}

size_t suffix_index(const std::string& name, const std::string& prefix) {
    if (name.rfind(prefix, 0) != 0) {
        fail("plugin compilation", "Unexpected cache input name: " + name);
    }
    try {
        return static_cast<size_t>(std::stoull(name.substr(prefix.size())));
    } catch (const std::exception&) {
        fail("plugin compilation", "Invalid cache input name: " + name);
    }
}

size_t op_count(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    const auto nodes = model->get_ordered_ops();
    return static_cast<size_t>(std::count_if(nodes.begin(),
                                             nodes.end(),
                                             [&](const std::shared_ptr<ov::Node>& node) {
                                                 return node->get_type_name() == type_name;
                                             }));
}

std::vector<std::shared_ptr<ov::Node>> paged_attention_nodes(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> result;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()).find("PagedAttention") != std::string::npos) {
            result.push_back(node);
        }
    }
    return result;
}

std::vector<std::shared_ptr<ov::Node>> nodes_of_type(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& type_name) {
    std::vector<std::shared_ptr<ov::Node>> result;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == type_name) {
            result.push_back(node);
        }
    }
    return result;
}

bool has_ancestor_parameter(const ov::Output<ov::Node>& output,
                            const std::string& name,
                            std::set<const ov::Node*>& seen) {
    const auto node = output.get_node_shared_ptr();
    if (!seen.insert(node.get()).second) {
        return false;
    }
    if (node->get_type_name() == std::string("Parameter") &&
        (node->get_friendly_name() == name || node->output(0).get_names().count(name) != 0)) {
        return true;
    }
    for (size_t input_index = 0; input_index < node->get_input_size(); ++input_index) {
        if (has_ancestor_parameter(node->input_value(input_index), name, seen)) {
            return true;
        }
    }
    return false;
}

void print_pa_contract(const std::vector<std::shared_ptr<ov::Node>>& pa_nodes) {
    for (size_t node_index = 0; node_index < pa_nodes.size(); ++node_index) {
        const auto& node = pa_nodes[node_index];
        std::cout << "PagedAttention[" << node_index << "] " << node->get_type_name() << " name="
                  << node->get_friendly_name() << '\n';
        for (size_t input_index = 0; input_index < node->get_input_size(); ++input_index) {
            const auto input = node->input_value(input_index);
            std::cout << "  input[" << input_index << "] type=" << input.get_element_type()
                      << " shape=" << partial_shape_string(input.get_partial_shape())
                      << " source=" << input.get_node()->get_friendly_name() << '\n';
        }
        for (size_t output_index = 0; output_index < node->get_output_size(); ++output_index) {
            const auto output = node->output(output_index);
            std::cout << "  output[" << output_index << "] type=" << output.get_element_type()
                      << " shape=" << partial_shape_string(output.get_partial_shape()) << '\n';
        }
    }
}

void print_sdpa_contract(const std::shared_ptr<ov::Model>& model) {
    const auto nodes = nodes_of_type(model, "ScaledDotProductAttention");
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const auto& node = nodes[node_index];
        std::cout << "SDPA[" << node_index << "] name=" << node->get_friendly_name() << '\n';
        for (size_t input_index = 0; input_index < node->get_input_size(); ++input_index) {
            const auto input = node->input_value(input_index);
            std::cout << "  input[" << input_index << "] type=" << input.get_element_type()
                      << " shape=" << partial_shape_string(input.get_partial_shape())
                      << " source=" << input.get_node()->get_friendly_name() << '\n';
        }
    }
}

void validate_raw_draft(const std::shared_ptr<ov::Model>& model, const Fixture& fixture) {
    const size_t layers = fixture.size("model_layers");
    const size_t sdpa = op_count(model, "ScaledDotProductAttention");
    const size_t reads = op_count(model, "ReadValue");
    const size_t assigns = op_count(model, "Assign");
    const size_t cache_state_ops = layers * 2;
    if (sdpa != layers || reads != cache_state_ops || assigns != cache_state_ops) {
        fail("fixture/model validation",
             "Expected " + std::to_string(layers) + " SDPA nodes and " +
                 std::to_string(cache_state_ops) + " ReadValue/Assign nodes, got SDPA=" +
                 std::to_string(sdpa) + ", ReadValue=" + std::to_string(reads) +
                 ", Assign=" + std::to_string(assigns));
    }
    try {
        const auto output = model->output("last_hidden_state");
        if (output.get_node()->get_type_name() == std::string("MatMul")) {
            fail("fixture/model validation", "Raw DFlash model unexpectedly exposes an lm-head MatMul output");
        }
        static_cast<void>(model->input("token_type_ids"));
    } catch (const ov::Exception& error) {
        fail("fixture/model validation", "Raw DFlash port validation failed: " + std::string(error.what()));
    }
}

void remove_orphan_parameters(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> unused;
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->output(0).get_target_inputs().empty()) {
            unused.push_back(parameter);
        }
    }
    for (const auto& parameter : unused) {
        model->remove_parameter(parameter);
    }
}

size_t max_fixture_full_length(const Fixture& fixture);

std::shared_ptr<ov::op::v0::Parameter> parameter_by_name(const std::shared_ptr<ov::Model>& model,
                                                         const std::string& name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == name || parameter->output(0).get_names().count(name) != 0) {
            return parameter;
        }
    }
    fail("fixture/model validation", "Raw DFlash model is missing parameter '" + name + "'");
}

void replace_parameter(const std::shared_ptr<ov::Model>& model,
                       const std::shared_ptr<ov::op::v0::Parameter>& parameter,
                       const ov::Output<ov::Node>& replacement) {
    std::vector<ov::Input<ov::Node>> consumers;
    for (const auto& consumer : parameter->output(0).get_target_inputs()) {
        consumers.push_back(consumer);
    }
    if (consumers.empty()) {
        fail("fixture/model validation",
             "Raw DFlash parameter '" + parameter->get_friendly_name() + "' has no consumers");
    }
    for (auto& consumer : consumers) {
        consumer.replace_source_output(replacement);
    }
    if (!parameter->output(0).get_target_inputs().empty()) {
        fail("fixture/model validation",
             "Failed to detach all consumers of raw DFlash parameter '" + parameter->get_friendly_name() + "'");
    }
    model->remove_parameter(parameter);
}

void assert_all_valid_attention_mask(const Fixture& fixture) {
    const size_t steps = fixture.size("steps");
    for (size_t step = 0; step < steps; ++step) {
        const auto& mask = fixture.tensor("step" + std::to_string(step) + ".attention_mask");
        if (mask.type != DataType::i64) {
            fail("fixture/model validation", "DFlash attention_mask fixture is not i64");
        }
        const size_t values = checked_product(mask.shape, mask.name);
        for (size_t index = 0; index < values; ++index) {
            int64_t value = 0;
            std::memcpy(&value, mask.bytes.data() + index * sizeof(value), sizeof(value));
            if (value != 1) {
                fail("fixture/model validation",
                     "The standalone no-padding simulator only supports all-valid attention masks");
            }
        }
    }
}

void specialize_single_sequence_controls(const std::shared_ptr<ov::Model>& model, const Fixture& fixture) {
    // The fixture is batch-one and contains no padding or beam reordering.
    // Freeze these stateful-SDPA-only controls before the PA pass so it cannot
    // remove them from the model parameter list while their old consumers
    // remain live.
    assert_all_valid_attention_mask(fixture);
    const size_t max_context_length = max_fixture_full_length(fixture);

    const auto beam_idx = parameter_by_name(model, "beam_idx");
    auto beam_idx_constant = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    beam_idx_constant->set_friendly_name("dflash_simulator_beam_idx_zero");
    replace_parameter(model, beam_idx, beam_idx_constant->output(0));

    const auto attention_mask = parameter_by_name(model, "attention_mask");
    std::vector<int64_t> all_valid(max_context_length, 1);
    auto attention_mask_constant =
        ov::op::v0::Constant::create(ov::element::i64, {1, max_context_length}, all_valid);
    attention_mask_constant->set_friendly_name("dflash_simulator_attention_mask_all_valid");
    replace_parameter(model, attention_mask, attention_mask_constant->output(0));
    model->validate_nodes_and_infer_types();
    std::cout << "Specialized batch-1 controls: beam_idx={0}, attention_mask=all-valid[1,"
              << max_context_length << "]\n";
}

std::vector<std::shared_ptr<ov::Node>> transform_to_paged_attention(const std::shared_ptr<ov::Model>& model,
                                                                     const Fixture& fixture) {
    try {
        ov::pass::SDPAToPagedAttention(
            /* use_per_layer_block_indices_inputs= */ false,
            /* use_score_outputs= */ false,
            /* allow_score_aggregation= */ true,
            /* allow_cache_rotation= */ false,
            /* allow_xattention= */ false,
            /* allow_adaptive_rkv= */ false,
            /* allow_qq_bias= */ false)
            .run_on_model(model);
        // The native pass can make state-only controls (beam_idx and the
        // disconnected SDPA mask) orphaned. They are not PA inputs and must
        // not remain declared model parameters.
        remove_orphan_parameters(model);
        model->validate_nodes_and_infer_types();
    } catch (const std::exception& error) {
        fail("PA transform",
             "SDPAToPagedAttention did not produce a valid DFlash PA graph before PA ABI validation. "
             "The transform/op path must be fixed before GPU kernel execution; the native error is: " +
                 std::string(error.what()));
    }

    const size_t layers = fixture.size("model_layers");
    const auto pa_nodes = paged_attention_nodes(model);
    if (pa_nodes.empty() && op_count(model, "ScaledDotProductAttention") == layers) {
        fail("transformed-model validation",
             "SDPAToPagedAttention completed after beam_idx/attention_mask specialization but matched no DFlash "
             "attention nodes. The remaining rank-5 asymmetric Q=B, K/V=D+B SDPA layout needs a dedicated "
             "transform matcher/PA ABI.");
    }
    if (op_count(model, "ScaledDotProductAttention") != 0 || op_count(model, "ReadValue") != 0 ||
        op_count(model, "Assign") != 0 || pa_nodes.size() != layers) {
        fail("transformed-model validation",
             "Expected all SDPA/ReadValue/Assign nodes to become " + std::to_string(layers) +
                 " PagedAttention nodes, got PA=" + std::to_string(pa_nodes.size()) +
                 ", SDPA=" + std::to_string(op_count(model, "ScaledDotProductAttention")) +
                 ", ReadValue=" + std::to_string(op_count(model, "ReadValue")) +
                 ", Assign=" + std::to_string(op_count(model, "Assign")));
    }
    for (const auto& node : pa_nodes) {
        if (node->get_input_size() <= 25) {
            fail("transformed-model validation",
                 "PagedAttention node '" + node->get_friendly_name() +
                     "' has no token_type_ids ABI input 25; got " + std::to_string(node->get_input_size()) +
                     " inputs");
        }
        std::set<const ov::Node*> seen;
        if (!has_ancestor_parameter(node->input_value(25), "token_type_ids", seen)) {
            fail("transformed-model validation",
                 "PagedAttention node '" + node->get_friendly_name() +
                     "' input 25 is not derived from token_type_ids");
        }
    }
    return pa_nodes;
}

struct LayerCache {
    ov::Tensor key;
    ov::Tensor value;
};

using LayerCaches = std::map<size_t, LayerCache>;

ov::Shape concrete_cache_shape(const ov::PartialShape& partial_shape, size_t blocks) {
    if (partial_shape.rank().is_dynamic() || partial_shape.rank().get_length() != 4) {
        fail("plugin compilation",
             "Expected rank-4 PagedAttention KV cache input, got " + partial_shape_string(partial_shape));
    }
    ov::Shape shape;
    for (size_t index = 0; index < 4; ++index) {
        if (index == 0) {
            shape.push_back(blocks);
            continue;
        }
        if (partial_shape[index].is_dynamic()) {
            fail("plugin compilation",
                 "KV cache non-block dimension is dynamic: " + partial_shape_string(partial_shape));
        }
        shape.push_back(static_cast<size_t>(partial_shape[index].get_length()));
    }
    return shape;
}

size_t max_fixture_full_length(const Fixture& fixture) {
    const size_t steps = fixture.size("steps");
    size_t maximum = 0;
    for (size_t step = 0; step < steps; ++step) {
        maximum = std::max(maximum, fixture.size("step." + std::to_string(step) + ".full_len"));
    }
    return maximum;
}

LayerCaches allocate_gpu_caches(ov::InferRequest& request,
                                const InputMap& inputs,
                                ov::RemoteContext& context,
                                size_t layers,
                                size_t physical_blocks,
                                size_t& cache_block_size) {
    LayerCaches caches;
    for (const auto& [name, input] : inputs) {
        const bool is_key = name.rfind("key_cache.", 0) == 0;
        const bool is_value = name.rfind("value_cache.", 0) == 0;
        if (!is_key && !is_value) {
            continue;
        }
        if (input.get_element_type() != ov::element::f16) {
            fail("plugin compilation",
                 "GPU KV cache input '" + name + "' is " + input.get_element_type().to_string() + ", not f16");
        }
        const ov::Shape shape = concrete_cache_shape(input.get_partial_shape(), physical_blocks);
        if (shape[2] != 16) {
            fail("plugin compilation",
                 "GPU KV cache input '" + name + "' has block size " + std::to_string(shape[2]) +
                     ", expected 16");
        }
        if (cache_block_size == 0) {
            cache_block_size = shape[2];
        } else if (cache_block_size != shape[2]) {
            fail("plugin compilation", "Key/value cache block sizes differ");
        }
        ov::Tensor remote = context.create_tensor(ov::element::f16, shape);
        request.set_tensor(name, remote);
        const size_t layer = suffix_index(name, is_key ? "key_cache." : "value_cache.");
        auto& cache = caches[layer];
        if (is_key) {
            cache.key = remote;
        } else {
            cache.value = remote;
        }
    }
    if (caches.size() != layers) {
        fail("plugin compilation",
             "Expected " + std::to_string(layers) + " key/value cache layer pairs, got " +
                 std::to_string(caches.size()));
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        const auto iterator = caches.find(layer);
        if (iterator == caches.end() || !iterator->second.key || !iterator->second.value) {
            fail("plugin compilation", "Incomplete cache pair for layer " + std::to_string(layer));
        }
    }
    return caches;
}

ov::Tensor int32_tensor(const ov::Output<const ov::Node>& port, const std::vector<int32_t>& values) {
    if (port.get_element_type() != ov::element::i32) {
        fail("inference",
             "Expected i32 PA metadata input '" + port_name(port) + "', got " + port.get_element_type().to_string());
    }
    const auto rank = port.get_partial_shape().rank();
    ov::Shape shape;
    if (rank.is_static() && rank.get_length() == 0) {
        if (values.size() != 1) {
            fail("inference", "Scalar PA metadata input requires exactly one value");
        }
    } else {
        shape = {values.size()};
    }
    ov::Tensor result(ov::element::i32, shape);
    std::copy(values.begin(), values.end(), result.data<int32_t>());
    return result;
}

void set_fixture_input(ov::InferRequest& request,
                       const InputMap& inputs,
                       const Fixture& fixture,
                       const std::string& step_prefix,
                       const std::string& model_name) {
    const auto iterator = inputs.find(model_name);
    if (iterator == inputs.end()) {
        return;
    }
    request.set_tensor(model_name,
                       materialize_input(fixture.tensor(step_prefix + "." + model_name),
                                         model_name,
                                         iterator->second.get_element_type(),
                                         iterator->second.get_partial_shape()));
}

void set_beam_index_if_needed(ov::InferRequest& request, const InputMap& inputs) {
    const auto iterator = inputs.find("beam_idx");
    if (iterator == inputs.end()) {
        return;
    }
    if (iterator->second.get_element_type() != ov::element::i32) {
        fail("inference", "beam_idx is not i32");
    }
    ov::Tensor beam_idx(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;
    request.set_tensor("beam_idx", beam_idx);
}

void set_pa_metadata(ov::InferRequest& request,
                     const InputMap& inputs,
                     size_t past_length,
                     size_t block_length,
                     size_t full_length,
                     size_t cache_block_size) {
    const auto set = [&](const std::string& name, const std::vector<int32_t>& values) {
        const auto iterator = inputs.find(name);
        if (iterator == inputs.end()) {
            fail("inference", "Transformed model is missing PA input '" + name + "'");
        }
        request.set_tensor(name, int32_tensor(iterator->second, values));
    };

    const size_t active_blocks = (full_length + cache_block_size - 1) / cache_block_size;
    std::vector<int32_t> block_indices(active_blocks);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    set("past_lens", {static_cast<int32_t>(past_length)});
    set("subsequence_begins", {0, static_cast<int32_t>(block_length)});
    set("block_indices", block_indices);
    set("block_indices_begins", {0, static_cast<int32_t>(active_blocks)});
    set("max_context_len", {static_cast<int32_t>(full_length)});
    if (inputs.count("score_aggregation_window") != 0) {
        set("score_aggregation_window", {1});
    }
}

std::vector<float> unpack_paged_cache(const ov::Tensor& remote_cache,
                                      size_t logical_length,
                                      size_t cache_block_size) {
    if (remote_cache.get_element_type() != ov::element::f16) {
        fail("inference", "Attempted to unpack non-FP16 GPU KV cache");
    }
    const ov::Shape shape = remote_cache.get_shape();
    if (shape.size() != 4 || shape[2] != cache_block_size) {
        fail("inference", "Unexpected GPU KV cache shape: " + shape_string(shape));
    }
    ov::Tensor host_cache(ov::element::f16, shape);
    remote_cache.copy_to(host_cache);

    const size_t heads = shape[1];
    const size_t head_dim = shape[3];
    const auto* source = host_cache.data<const ov::float16>();
    std::vector<float> result(heads * logical_length * head_dim);
    for (size_t head = 0; head < heads; ++head) {
        for (size_t token = 0; token < logical_length; ++token) {
            const size_t block = token / cache_block_size;
            const size_t within_block = token % cache_block_size;
            for (size_t channel = 0; channel < head_dim; ++channel) {
                const size_t source_index =
                    ((block * heads + head) * cache_block_size + within_block) * head_dim + channel;
                const size_t destination_index = (head * logical_length + token) * head_dim + channel;
                result[destination_index] = static_cast<float>(source[source_index]);
            }
        }
    }
    return result;
}

void compare_named(const std::string& name,
                   const std::vector<float>& actual,
                   const RawTensor& expected,
                   const Options& options) {
    const auto comparison = compare(actual, raw_to_floats(expected), options.atol, options.rtol);
    print_comparison(name, comparison, actual.size());
    if (comparison.failures != 0) {
        fail("numerical comparison",
             name + " exceeded tolerances (atol=" + std::to_string(options.atol) +
                 ", rtol=" + std::to_string(options.rtol) + ")");
    }
}

void verify_model_checksums(const fs::path& xml, const Fixture& fixture) {
    const fs::path bin = xml.parent_path() / (xml.stem().string() + ".bin");
    const fs::path config = xml.parent_path() / "config.json";
    const auto verify = [&](const fs::path& path, const std::string& key) {
        const auto iterator = fixture.metadata.find(key);
        if (iterator == fixture.metadata.end()) {
            fail("fixture/model validation", "Fixture is missing checksum " + key);
        }
        const std::string actual = sha256_file(path);
        if (actual != iterator->second) {
            fail("fixture/model validation",
                 "Checksum mismatch for " + path.string() + ": fixture=" + iterator->second + ", actual=" + actual);
        }
    };
    verify(xml, "model_xml_sha256");
    verify(bin, "model_bin_sha256");
    verify(config, "model_config_sha256");
}

void validate_weight_precision(const std::shared_ptr<ov::Model>& model) {
    size_t fp16_elements = 0;
    size_t quantized_weight_elements = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() != std::string("Constant")) {
            continue;
        }
        const size_t elements = node->output(0).get_shape().empty()
                                    ? 1
                                    : checked_product(node->output(0).get_shape(), node->get_friendly_name());
        if (node->output(0).get_element_type() == ov::element::f16) {
            fp16_elements += elements;
        }
        if (node->output(0).get_element_type() == ov::element::u8 ||
            node->output(0).get_element_type() == ov::element::i8 ||
            node->output(0).get_element_type() == ov::element::u4 ||
            node->output(0).get_element_type() == ov::element::i4) {
            quantized_weight_elements += elements;
        }
    }
    if (fp16_elements == 0 || quantized_weight_elements != 0) {
        fail("fixture/model validation",
             "Expected FP16-weight DFlash model but found fp16_constant_elements=" + std::to_string(fp16_elements) +
                 ", quantized_constant_elements=" + std::to_string(quantized_weight_elements));
    }
}

void validate_no_unknown_inputs(const InputMap& inputs) {
    static const std::set<std::string> known = {
        "inputs_embeds",
        "hidden_states",
        "position_ids",
        "attention_mask",
        "token_type_ids",
        "beam_idx",
        "past_lens",
        "subsequence_begins",
        "block_indices",
        "block_indices_begins",
        "max_context_len",
        "score_aggregation_window",
    };
    std::vector<std::string> unexpected;
    for (const auto& [name, _] : inputs) {
        if (!is_cache_name(name) && known.count(name) == 0) {
            unexpected.push_back(name);
        }
    }
    if (!unexpected.empty()) {
        fail("plugin compilation",
             "Transformed model has unsupported input(s): " + join(unexpected, ", "));
    }
}

void run(const Options& options) {
    const Fixture fixture = load_fixture(options.fixture);
    const fs::path model_xml = resolve_model_xml(options.model);
    verify_model_checksums(model_xml, fixture);

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(model_xml);
    } catch (const std::exception& error) {
        fail("fixture/model validation", "Cannot read OpenVINO model: " + std::string(error.what()));
    }
    validate_weight_precision(model);
    validate_raw_draft(model, fixture);
    if (options.dump_pa_ir) {
        print_sdpa_contract(model);
    }
    specialize_single_sequence_controls(model, fixture);
    std::cout.flush();

    auto pa_nodes = transform_to_paged_attention(model, fixture);
    if (options.dump_pa_ir) {
        print_pa_contract(pa_nodes);
    }

    ov::AnyMap properties;
    properties[ov::hint::inference_precision.name()] = ov::element::f16;
    properties[ov::hint::kv_cache_precision.name()] = ov::element::f16;

    ov::CompiledModel compiled;
    try {
        compiled = core.compile_model(model, options.device, properties);
    } catch (const std::exception& error) {
        fail("GPU plugin compilation",
             "GPU compilation failed after successful PA transformation: " + std::string(error.what()));
    }

    const auto execution_devices = compiled.get_property(ov::execution_devices);
    std::cout << "OpenVINO version: " << ov::get_openvino_version() << '\n';
    std::cout << "Requested device: " << options.device << '\n';
    std::cout << "Execution devices: " << join(execution_devices, ", ") << '\n';
    std::cout << "Requested inference/KV cache precision: f16/f16\n";

    const InputMap inputs = inputs_by_name(compiled);
    validate_no_unknown_inputs(inputs);
    ov::InferRequest request = compiled.create_infer_request();
    ov::RemoteContext context = compiled.get_context();
    if (!context || context.get_device_name().rfind("GPU", 0) != 0) {
        fail("GPU plugin compilation", "Compiled model did not provide a GPU remote context");
    }

    size_t cache_block_size = 0;
    const size_t max_length = max_fixture_full_length(fixture);
    const size_t physical_blocks = (max_length + 15) / 16;
    const size_t layers = fixture.size("model_layers");
    const LayerCaches caches =
        allocate_gpu_caches(request, inputs, context, layers, physical_blocks, cache_block_size);

    const size_t steps = fixture.size("steps");
    for (size_t step = 0; step < steps; ++step) {
        const std::string prefix = "step" + std::to_string(step);
        const size_t past_length = fixture.size("step." + std::to_string(step) + ".past_len");
        const size_t delta_length = fixture.size("step." + std::to_string(step) + ".delta_len");
        const size_t block_length = fixture.size("step." + std::to_string(step) + ".block_len");
        const size_t full_length = fixture.size("step." + std::to_string(step) + ".full_len");
        const size_t committed_length = fixture.size("step." + std::to_string(step) + ".committed_len");
        if (full_length != past_length + delta_length + block_length || committed_length != past_length + delta_length) {
            fail("fixture/model validation", "Invalid DFlash cache-length arithmetic for step " + std::to_string(step));
        }

        set_fixture_input(request, inputs, fixture, prefix, "inputs_embeds");
        set_fixture_input(request, inputs, fixture, prefix, "hidden_states");
        set_fixture_input(request, inputs, fixture, prefix, "position_ids");
        set_fixture_input(request, inputs, fixture, prefix, "attention_mask");
        set_fixture_input(request, inputs, fixture, prefix, "token_type_ids");
        set_beam_index_if_needed(request, inputs);
        set_pa_metadata(request, inputs, past_length, block_length, full_length, cache_block_size);

        try {
            request.infer();
        } catch (const std::exception& error) {
            fail("GPU inference",
                 "GPU inference failed for DFlash step " + std::to_string(step) + ": " + std::string(error.what()));
        }

        std::cout << "Step " << step << " (past=" << past_length << ", D=" << delta_length
                  << ", B=" << block_length << ", full=" << full_length << ")\n";
        try {
            const auto output = request.get_tensor("last_hidden_state");
            compare_named(prefix + ".last_hidden_state",
                          tensor_to_floats(output),
                          fixture.tensor(prefix + ".expected.last_hidden_state"),
                          options);
        } catch (const ov::Exception& error) {
            fail("GPU inference", "Cannot retrieve last_hidden_state: " + std::string(error.what()));
        }

        for (size_t layer = 0; layer < layers; ++layer) {
            const auto cache = caches.at(layer);
            compare_named(prefix + ".full.key." + std::to_string(layer),
                          unpack_paged_cache(cache.key, full_length, cache_block_size),
                          fixture.tensor(prefix + ".expected.key." + std::to_string(layer)),
                          options);
            compare_named(prefix + ".full.value." + std::to_string(layer),
                          unpack_paged_cache(cache.value, full_length, cache_block_size),
                          fixture.tensor(prefix + ".expected.value." + std::to_string(layer)),
                          options);
            compare_named(prefix + ".committed.key." + std::to_string(layer),
                          unpack_paged_cache(cache.key, committed_length, cache_block_size),
                          fixture.tensor(prefix + ".committed.key." + std::to_string(layer)),
                          options);
            compare_named(prefix + ".committed.value." + std::to_string(layer),
                          unpack_paged_cache(cache.value, committed_length, cache_block_size),
                          fixture.tensor(prefix + ".committed.value." + std::to_string(layer)),
                          options);
        }

        const size_t full_blocks = (full_length + cache_block_size - 1) / cache_block_size;
        const size_t committed_blocks = (committed_length + cache_block_size - 1) / cache_block_size;
        std::cout << "  cache lifecycle: active blocks " << full_blocks << " -> " << committed_blocks
                  << ", released full temporary blocks=" << (full_blocks - committed_blocks) << '\n';
    }

    std::cout << "PASS: GPU FP16 DFlash PA fixture matched PyTorch reference.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse_options(argc, argv));
        return 0;
    } catch (const StageError& error) {
        std::cerr << "FAIL [" << error.stage() << "]: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "FAIL [unexpected]: " << error.what() << '\n';
        return 3;
    }
}
