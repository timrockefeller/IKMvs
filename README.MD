#### Usage

复制到项目根目录。

#### Documents

##### IK_InitProject(PROJECT_NAME_STR)

初始化项目，`PROJECT_NAME_STR`为项目名称。

##### IK_AddAllSubDir()

将所有子目录加入项目处理过程。

##### IK_SetupProject(MODE TARGET_NAME STR_TARGET_SOURCES STR_TARGET_LIBS)

生成单个目标文件。

- `MODE`：目标文件类型("LIB"|"BIN")
- `TARGET_NAME`：目标文件名称
- `STR_TARGET_SOURCES`：源文件，若空请留一个空格
- `STR_TARGET_LIBS`：链接库，若空请留一个空格

##### IK_UnityArgs(<TARGET_STR> [arg1 [arg2 [...]]])

创建变量以适配IK_SetupProject。

##### Anything else when using CMake
