"""The generation step: turning a prompt into an answer with citations.

This is the last stage of the pipeline.  It takes the prompt the
augmentation step assembled - the question, the retrieved passages and the
conversation so far - and asks the model to answer it.  The answer comes
back token by token, and the citation markers the model used are resolved
back to the passages they point at, so the user can see where each claim
came from.
"""
