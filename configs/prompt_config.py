# prompt模板使用Jinja2语法，简单点就是用双大括号代替f-string的单大括号
# 本配置文件支持热加载，修改prompt模板后无需重启服务。

# LLM对话支持的变量：
#   - input: 用户输入内容

# 知识库支持的变量：
#   - context: 从检索结果拼接的知识文本
#   - question: 用户提出的问题

PROMPT_TEMPLATES = {
    "llm_chat": {
        "default":
            '{{ input }}',
    },
    "knowledge_base_chat": {
        "default":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',
    }
}
