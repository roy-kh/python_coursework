# Roy Hayyat, Hayyat@usc.edu
# ITP 115, Spring 2021
# Final Project

from MenuItem import MenuItem  # import statement


class Menu:  # class represents restaurant’s menu containing four categories of menu items diners can order from.
    CATEGORIES = ["Drink", "Appetizer", "Entree", "Dessert"]  # a list containing 4 strings, representing the
    # 4 possible types of menu items. Acts as a static variable.

    def __init__(self, menu="menu.csv"):  # The class will use the following instance attribute:
        # Parameter (1): a string representing the name of the CSV (comma separated values) file that contains
        # information about the menu items for the restaurant’s menu
        self.drinks = []  # a list containing MenuItem objects that are the drink menu items from the menu
        self.appetizers = []  # a list containing MenuItem objects that are the appetizer menu items from the menu
        self.entrees = []  # a list containing MenuItem objects that are the entree menu items from the menu
        self.desserts = []  # a list containing MenuItem objects that are the dessert menu items from the menu
# Define the following methods:
        fin = open(menu, "r")  # Open and read the CSV file
        for line in fin:  # create a MenuItem object from each line in the file.
            line = line.strip()
            splitMenu = line.split(",")  # Use the type to add the new object to one of the instance attributes.
            menuItem = MenuItem(splitMenu[0], splitMenu[1], splitMenu[2], splitMenu[3])  # Note that each line in
            # the file contains the 4 pieces of information needed to create a MenuItem object.
            if splitMenu[1] == "Drink":
                self.drinks.append(menuItem)
            elif splitMenu[1] == "Appetizer":
                self.appetizers.append(menuItem)
            elif splitMenu[1] == "Entree":
                self.entrees.append(menuItem)
            elif splitMenu[1] == "Dessert":
                self.desserts.append(menuItem)
        fin.close()  # Close the file object.

    def getMenuItem(self, category, itemPosition):
        """
        :param category: string representing category
        :param itemPosition: integer representing index position of menu item
        :return: a MenuItem object from the appropriate instance attribute
        """
        if category in Menu.CATEGORIES:  # For error checking, make sure that the category parameter
            # is one of the CATEGORIES.
            if category == "Drink":
                if 0 <= itemPosition < len(self.drinks):
                    return self.drinks[itemPosition]
            elif category == "Appetizer":
                if 0 <= itemPosition < len(self.appetizers):
                    return self.appetizers[itemPosition]
            elif category == "Entree":
                if 0 <= itemPosition < len(self.entrees):
                    return self.entrees[itemPosition]
            elif category == "Dessert":
                if 0 <= itemPosition < len(self.desserts):
                    return self.desserts[itemPosition]

        # if 0 <= itemPosition < len(Menu.CATEGORIES):  # For error checking, make sure that the index parameter    (Do I put this outside of the other loop?)
        #     # is in the range of the appropriate list.
        #     return  # Return the correct MenuItem using its type and index position.
        # else:
        #     return False  # If the category and/or index were not valid, then do not return anything.

    def printMenuItems(self, category):
        """
        :param category: a string representing a category
        :return: None
        """
        counter = 0
        if category in Menu.CATEGORIES:  # For error checking, make sure that the category parameter is
        # one of the CATEGORIES. If it is not, then don’t print anything.
            print("-----" + category.upper() + "S" + "-----")  # Print a header with the type of menu items, followed by a numbered list of the menu items of that category.
            if category == "Drink":
                for item in self.drinks:
                    print(str(counter) + ")", item)
                    counter += 1
            if category == "Appetizer":
                for item in self.appetizers:
                    print(str(counter) + ")", item)
                    counter += 1
            if category == "Entree":
                for item in self.entrees:
                    print(str(counter) + ")", item)
                    counter += 1
            if category == "Dessert":
                for item in self.desserts:
                    print(str(counter) + ")", item)
                    counter += 1

    def getNumMenuItems(self, category):
        """
        :param category: a string representing a category
        :return: integer representing the number of MenuItems within the category given
        """
        if category in Menu.CATEGORIES:  # make sure that the category parameter is one of the CATEGORIES.
            if category == "Drink":
                return len(self.drinks)
            elif category == "Appetizer":
                return len(self.appetizers)
            elif category == "Entree":
                return len(self.entrees)
            elif category == "Dessert":
                return len(self.drinks)
        else:  # If not, then return 0.
            return 0

# Unit test
# def main():
#     menuObj = Menu("menu.csv")
#     for category in Menu.CATEGORIES:
#         menuObj.printMenuItems(category)
#
#
# main()
