# 知识问答模板
#   - context: 从检索结果拼接的知识文本
#   - question: 用户提出的问题
KNOWLEDGE_CHAT_TEMPLATE = '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，' \
                          '不允许在答案中添加编造成分，答案请使用中文。 </指令>\n' \
                          '<已知信息>{${context}}</已知信息>\n' \
                          '<问题>{${question}}</问题>\n'


# 提取query中关键信息模板
# - user_request: 用户问题
EXTRACT_INFO_TEMPLATE = """请提取问题中的可以帮助检索的重点信息片段和任务描述，以JSON的格式给出：{{"information": ["重点信息片段1", "重点信息片段2"], "instruction": ["任务描述片段1", "任务描述片段2"]}}。
如果是提问，则默认任务描述为：回答问题

Question: MMDET.UTILS是什么
Result: {{"information": ["MMDET.UTILS是什么"], "instruction": ["回答问题"]}}
Observation: ...

Question: 总结
Result: {{"information": [], "instruction": ["总结"]}}
Observation: ...

Question: 要非常详细描述2.1 DATA，2.2 TOKENIZATION，2.3 ARCHITECTURE。另外你能把这篇论文的方法融合进去吗
Result: {{"information": ["2.1 DATA，2.2 TOKENIZATION，2.3 ARCHITECTURE"], "instruction": ["要非常详细描述", "另外你能把这篇论文的方法融合进去吗"]}}
Observation: ...

Question: 帮我统计不同会员等级的业绩
Result: {{"information": ["会员等级的业绩"], "instruction": ["帮我统计"]}}
Observation: ...

Question: ${user_request}
Result:
"""