const nextJest = require("next/jest");

const createJestConfig = nextJest({
  dir: "./",
});

const customJestConfig = {
  testEnvironment: "node",
  testMatch: ["**/*.test.ts"],
};

module.exports = createJestConfig(customJestConfig);
