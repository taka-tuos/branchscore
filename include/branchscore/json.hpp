#pragma once

#include <map>
#include <string>
#include <vector>

namespace branchscore::json {

class Value {
public:
    enum class Type {
        null_value,
        boolean,
        number,
        string,
        array,
        object,
    };

    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value();
    explicit Value(bool value);
    explicit Value(double value);
    explicit Value(std::string value);
    explicit Value(const char * value);
    explicit Value(Array value);
    explicit Value(Object value);

    Type type() const noexcept;
    bool boolean() const;
    double number() const;
    const std::string & string() const;
    const Array & array() const;
    const Object & object() const;
    Object & object();

    const Value * find(const std::string & key) const noexcept;
    void set(std::string key, Value value);

private:
    Type type_ = Type::null_value;
    bool boolean_ = false;
    double number_ = 0.0;
    std::string string_;
    Array array_;
    Object object_;
};

Value parse(const std::string & text);
std::string stringify(const Value & value);

} // namespace branchscore::json
