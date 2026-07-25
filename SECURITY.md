# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v1.0.x  | Yes (active dev)   |
| v0.4.x  | Security fixes only|
| < v0.4  | No                 |

## Reporting Vulnerabilities

Report security issues privately to the maintainer. Do not open public issues for security vulnerabilities.

## Model Security Considerations

As an open-source foundation model project, we recognize risks:
- Prompt injection attempts on served models.
- Model weight misuse.
- Data poisoning in training pipelines.

We design defensive measures (filtering, validation, red-teaming) as part of the development pipeline, not as afterthoughts.
