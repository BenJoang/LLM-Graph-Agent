# Skill_tool

## DESCRIPTION
按 skill 名称读取项目 skills 目录下对应技能的 skill.md 或 SKILL.md 内容。
## PROMPT

### WHEN_TO_USE
当用户要求使用、查看、加载、测试某个 skill，或给出 skill 名称时，使用这个工具读取该 skill 的说明文件。

### WHEN_NOT_TO_USE
不要用它读取任意文件。它只读取 skills/<skill_name>/skill.md。

### INPUT_RULES
- `skill_name` 必须是 skills 目录下的目录名（例如 `data-analyzer`）。
- 不要传入文件路径。
- 不要传入 `skills/<skill_name>`（带目录前缀）。
- 不要传入 `<skill_name>.md`（带扩展名）。

### LIMITS
- 工具只读取项目根目录 `skills/<skill_name>/skill.md`（或 `SKILL.md`）。
- 如果 skill 目录不存在或 skill.md 不存在，工具会失败。
- 不要改用 read_file 猜测 `skills/<skill_name>.md` 路径。
