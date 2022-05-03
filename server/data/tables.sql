DROP TABLE IF EXISTS defaults;
DROP TABLE IF EXISTS inputs;
DROP TABLE IF EXISTS params;
DROP TABLE IF EXISTS actions;
CREATE TABLE actions (
    name VARCHAR(20) PRIMARY KEY,
    info VARCHAR(500) NOT NULL,
    link VARCHAR(200),
    is_attack BOOlEAN NOT NULL
);
CREATE TABLE params (
    name VARCHAR(20) PRIMARY KEY,
    info VARCHAR(500) NOT NULL,
    lower_bound FLOAT,
    upper_bound FLOAT,
    type VARCHAR(200) NOT NULL
);
CREATE TABLE inputs (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    input_type VARCHAR(20)
);
CREATE TABLE defaults (
    action_id VARCHAR(20) REFERENCES actions(name),
    param_id VARCHAR(20) REFERENCES params(name),
    value FLOAT
);