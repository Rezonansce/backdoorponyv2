import importlib
import pkgutil
import sys

import backdoorpony

def get_as_package(package, debug=False):
    '''
    Imports the required package and returns its module of type ModuleType.

    Parameters
    ----------
    package (required) : str | module (i.e. module name | package)
        Contains the package to import.
    debug : Boolean
        Used for debugging purposes to print intermediate statements.

    Returns
    ----------
    package : module (actual import, type ModuleType)
    '''
    if isinstance(package, str):
        # Import the package
        package = importlib.import_module(package)
        if debug:
            print('PACKAGE: {0}'.format(str(package)))
    return package


def import_submodules_attributes(package, req_attr, result, imports={}, req_module=None, recursive=True, debug=False):
    '''
    Function that imports all submodules of a module, recursively, including subpackages.
    Retrieves the requested attributes for the given module,or for all modules in a package,
    depending on the value of req_module.
    Slices attributes that start with and end in two underscores for a cleaner return value.
    (i.e. for the returned result, the key will become 'attr' instead of '__attr__').

    Parameters
    ----------
    package (required) : str | module (i.e. module name | package)
        Contains the package to start looking from.
    req_attr (required) : list[str]
        Contains the requested attributes to return from the module/s.
    result (required): list[]
        In the beginning, result has to be an empty list to avoid appending to old results.
        Used to append the results throughout the recursion.
    imports : dict[]
        In the beginning, imports is an empty dictionary. Used to update the results throughout the recursion.
    req_module : str
        Contains the name of the module to look for; if None, all modules will be considered.
    recursive : Boolean
        Allows for applying recursion.
    debug : Boolean
        Used for debugging purposes to print intermediate statements.

    Returns
    ----------
    imports : dict[str, types.ModuleType]
        Dictionary with imported modules names as keys and actual module imports as values.
    result : list[dict[str(attribute_name), attribute_value)] (list stays empty if not found)
        One dictionary contains keys named after each of the respective module's attributes with
        their corresponding value.
        When there is one requested module, it contains one dictionary with the requested
        attributes as keys and their corresponding values.
        When the requested module is set to None, it contains multiple dictionaries for all
        found modules with the requested attributes as keys and their corresponding values.

    rtype : tuple [ dict[str, types.ModuleType], list[dict[str(attribute_name), attribute_value)] (list stays empty if not found)] ]
    '''

    # Imports the package
    package = get_as_package(package, debug)
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        # Full package name
        full_name = package.__name__ + '.' + name
        if debug:
            print('NAME: ' + str(name) + ' FULL NAME: ' + str(full_name))
        # Import subpackages or modules
        imports[full_name] = importlib.import_module(full_name)
        # If package, recursively call the method on it
        if recursive and is_pkg:
            imp, _ = import_submodules_attributes(
                package=full_name, req_attr=req_attr, imports=imports, req_module=req_module, result=result, recursive=recursive, debug=debug)
            imports.update(imp)
        # If not package, then get requested module attribute
        if not is_pkg:
            if debug:
                print('NAME: ' + str(name) + ' FULL NAME: ' + str(full_name))
            # Retrieves the imported module
            module = sys.modules[full_name]
            # If found req_module or if the requested is None (so it loops through all modules to add to the result), then look for attributes

            if req_module == name or req_module == None:
                module_object = {}
                # Loop through all the required attributes
                for attr in req_attr:
                    if hasattr(module, attr):
                        # Get the actual requested attribute
                        get_attr = getattr(module, attr)
                        # Removes the underscored of the attributes to return a cleaner object
                        if attr.startswith('__') and attr.endswith('__'):
                            sliced_attribute = attr[2:len(attr)-2]
                        module_object[sliced_attribute] = get_attr
                # If the requested attributes have been found, add the entire dictionary of attributes for this module to the result.
                # Prevents empty dictionaries from being added to the result if the requested attributes do not exist in the module.

                if module_object:
                    result.append(module_object)

    return imports, result


if __name__ == '__main__':
    imports, mods = import_submodules_attributes(package=backdoorpony.metrics, req_attr=[
                                                 '__category__', '__info__'], result=[])
    print('------------------------')
    print('IMPORTS: ' + str(imports) + '\n')
    print('MODULES ATTRIBUTES: ' + str(mods))