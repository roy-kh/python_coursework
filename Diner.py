# Roy Hayyat, Hayyat@usc.edu
# ITP 115, Spring 2021
# Final Project

from MenuItem import MenuItem  # import statement


class Diner:  # This class represents a diner at the restaurant and keeps tracks of their status and meal.
    STATUSES = ["seated", "ordering", "eating", "paying", "leaving"]  # a list containing the possible statuses
    # a diner might have. Acts as a static variable.

    def __init__(self, dinerName):  # The class will use the following instance attributes:
        """
        :param dinerName: This is the string representing the diner's name
        :return None
        """
        self.name = dinerName  # a string representing the diner’s name. Set to the input value.
        self.order = []  # a list of the MenuItem objects ordered by the diner
        self.status = 0  # an integer corresponding the diner’s current dining status

    # Define get methods for all the instance attributes.
    def getName(self):
        return self.name

    def getOrder(self):
        return self.order

    def getStatus(self):
        return self.status

    def updateStatus(self):
        """
        :param: None
        :return: None
        """
        self.status += 1  # Increases the diner’s status (instance attribute) by 1.

    def addToOrder(self, MenuItemObj):
        """
        :param MenuItemObj: this is a menuItem object
        :return: None
        """
        self.order.append(MenuItemObj)  # Appends the menu item to the end of the list of menu items (order
        # instance attribute).

    def printOrder(self):
        """
        :param: None
        :return: None
        """
        print(self.name + " ordered:")  # Print a message containing all the menu items the diner ordered.
        for item in self.order:  # Use the print() or str() functions to call the __str__ of the MenuItem class.
            print("- ", item)

    def getMealCost(self):
        """
        :param: None
        :return: float representing total cost of diner's meal
        """
        mealCostSum = 0.00
        for item in self.order:
            mealCostSum = item.getPrice() + mealCostSum  # Total up the cost of each of the menu items
            # the diner ordered.
        return mealCostSum

    def __str__(self):
        """
        :param: None
        :return: string
        """
        return "Diner " + self.name + " is currently " + Diner.STATUSES[self.status]  # Construct a message containing
        # the diner’s name and status, formatted in a readable manner. Use the class and instance attributes or methods.

# Unit test
# def main():
#     diner = Diner("Mehr")
#     diner.updateStatus()  # ordering
#     print(diner)
#     menuItem = MenuItem("Apple Pie", "Dessert", 7.0, "Cinnamon apples on our flaky crust.")
#     diner.addToOrder(menuItem)
#     diner.printOrder()
#     diner.updateStatus()  # eating
#     print(diner)
#     diner.updateStatus()  # paying
#     print(diner)
#     cost = diner.getMealCost()
#     print("cost = $" + str(cost))
#     diner.updateStatus()  # leaving
#     print(diner)
#
#
# main()
