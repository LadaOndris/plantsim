class NotImplementedException : public std::exception {

public:
    NotImplementedException(const char *error = "Functionality not yet implemented!") {
        errorMessage = error;
    }

    const char *what() const noexcept {
        return errorMessage.c_str();
    }

private:
    std::string errorMessage;
};