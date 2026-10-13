"""Project-configured separation of constituent parts and connection fasteners."""
from core.instance_category import instance_category


def slot_fastener(spec, model):
    return spec.get('fastener_by_model', {}).get(model) or spec.get('fastener_class')


def excluded_components(profile):
    names = set(profile.get('assembly_slot_managed_components', []))
    names.update(profile.get('assembly_forbidden_components', []))
    for spec in profile.get('assembly_interfaces', []):
        if spec.get('fastener_class'):
            names.add(spec['fastener_class'])
        names.update(spec.get('fastener_by_model', {}).values())
    return {instance_category(name, profile) for name in names if isinstance(name, str) and name}


def excluded_selections(components, profile):
    excluded = excluded_components(profile)
    return [name for name in components if instance_category(name, profile) in excluded]


def component_options(profile, states=()):
    names = set(profile.get('assembly_components', []))
    names.update(name for state in states for name in state.get('components', []))
    excluded = excluded_components(profile) | {'assembly', 'hand', 'left_hand', 'right_hand'}
    return sorted(name for name in names if isinstance(name, str) and name
                  and instance_category(name, profile) not in excluded)
