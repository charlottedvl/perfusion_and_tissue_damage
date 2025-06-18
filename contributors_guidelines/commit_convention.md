# Commit Convention

This repository follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) convention for commit messages.

The goal of this convention is to introduce a consistent format for commit messages, allowing them to be categorized more effectively.

## Format

The commit message should follow this structure:

```
<type>[optional scope]: <description>
[optional body]
[optional footer including BREAKING CHANGE footer]
```

## Description Guidelines

Many editors limit the length of the commit message summary.
Therefore, the description should be **concise yet precise** enough to clearly convey the purpose of the change.

### Vague Example:

```
docs: updating README.md
```

### Improved Example:

```
docs(containers): add instructions to build containers in README
```

## Common Commit Types

* `feat`: implement a new feature
* `fix`: fix a bug
* `docs`: add or update documentation
* `test`: add or update tests
* `refactor`: refactor one or more files (without adding features or fixing bugs)
* `clean`: remove unused files or eliminate unnecessary comments
* `perf`: improve application performance
* `style`: format code (e.g., spacing, indentation) with no functional change
* `build`: update build system or scripts
* `merge`: merge branches
* `deps`: add, remove, or update dependencies
* `config`: update configuration files (e.g., `.env`)

## Breaking Changes

Use the `BREAKING CHANGE` footer (or the `!` notation) to indicate changes that may **break backward compatibility** or significantly alter the behavior of the software.

### Example:

```
feat(cli)!: rename --output to --dest

BREAKING CHANGE: The `--output` flag in the CLI tool has been renamed to `--dest`.
Update all scripts or commands using this flag
```

## Best Practices

* Each commit should address **a single purpose or change**. Avoid combining unrelated changes in a single commit.
* Commit **frequently** to keep the project history clean and traceable.
* Keeping commits atomic and focused ensures easier debugging, code review, and reverts if needed.
