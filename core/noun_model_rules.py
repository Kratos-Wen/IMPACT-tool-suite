"""Project-owned model compatibility; no dataset classes are embedded here."""
from core.annotation_migration import model_from_trial
from core.instance_category import instance_category


def forbidden_nouns(trial, profile):
    model=model_from_trial(trial)
    rule=profile.get('model_component_rules',{}).get(model,{})
    return {instance_category(name,profile) for name in rule.get('forbidden_components',[])+rule.get('forbidden_nouns',[])}


def noun_allowed(name, trial, profile):
    return instance_category(name,profile) not in forbidden_nouns(trial,profile)
