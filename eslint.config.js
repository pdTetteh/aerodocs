export default [
  {
    files: ["app/ui/static/**/*.js", "tests/ui/**/*.js", "scripts/**/*.mjs"],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      globals: {
        AbortController: "readonly",
        Blob: "readonly",
        document: "readonly",
        fetch: "readonly",
        navigator: "readonly",
        Response: "readonly",
        TextDecoder: "readonly",
        window: "readonly",
      },
    },
    rules: {
      "no-console": "error",
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      "no-script-url": "error",
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "prefer-const": "error",
    },
  },
];
