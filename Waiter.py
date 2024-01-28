# Roy Hayyat, Hayyat@usc.edu
# ITP 115, Spring 2021
# Final Project

from Menu import Menu
from Diner import Diner  # import statements.


class Waiter:
    #  This class will represent the restaurant’s waiter. The waiter maintains a list of the diners it is currently
    #  taking care of, and progresses them through the different stages of the restaurant.
    #  The waiter in the simulation will repeat multiple cycles of attending to the diners. In each cycle,
    #  the waiter will seat a new diner, if one arrives, take any diners’ orders if needed, and give diners their bill,
    #  according to each diner’s status.

    def __init__(self, menuObject):  # The class will use the following instance attributes:
        """
        :param menuObject: a Menu object representing the restaurant's menu
        :return: None
        """
        self.diners = []  # a list of Diner objects the waiter is attending to
        self.menu = menuObject  # a Menu object representing the restaurant’s menu

    def addDiner(self, dinerObject):
        """
        :param dinerObject:
        :return: None
        """
        self.diners.append(dinerObject)  # Add the new Diner object to the waiter’s list of diners (instance attribute).

    def getNumDiners(self):
        """
        :param: None
        :return: integer representing the number of diners the waiter is keeping track of
        """
        return len(self.diners)

    def printDinerStatuses(self):
        """
        :param: None
        :return: None
        """
        # for diner in self.diners:  # Print all the diners the waiter is keeping track of, grouped by their statuses.
        #     if diner.getStatus() == 0:  # Loop through each of the possible dining statuses a Diner
        #         # might have by using the Diner class attribute to group the diners.
        #         print("Diners who are seated:")
        #         print("\n\tDiner " + Diner.getName(diner) + " is currently seated.")
        #     elif diner.getStatus() == 1:
        #         print("Diners who are ordering:"
        #               "\n\tDiner " + Diner.getName(diner) + " is currently ordering.")
        #     elif diner.getStatus() == 2:
        #         print("Diners who are eating:"
        #               "\n\tDiner " + Diner.getName(diner) + " is currently eating.")
        #     elif diner.getStatus() == 3:
        #         print("Diners who are paying:"
        #               "\n\tDiner " + Diner.getName(diner) + " is currently paying.")
        #     elif diner.getStatus() == 4:
        #         print("Diners who are leaving:"
        #               "\n\tDiner " + Diner.getName(diner) + " is currently leaving.")

        for status in Diner.STATUSES:
            print("Diners who are " + status + ":")
            for diner in self.diners:
                if diner.getStatus() == Diner.STATUSES.index(status):
                    # Loop through each of the possible dining statuses a Diner
                    # might have by using the Diner class attribute to group the diners.
                    print("\t", diner)
        print()

    def takeOrders(self):
        """
        :param: None
        :return: None
        """
        for diner in self.diners:  # Loop through the list of diners
            if diner.getStatus() == 1:  # check if the diner’s status is “ordering”.
                # For each diner that is ordering, loop through the different menu categories by using the
                # class attribute from the Menu class.
                for menuCategory in Menu.CATEGORIES:
                    self.menu.printMenuItems(menuCategory)  # For each category, print the menu items by calling
                    # the appropriate method in the Menu class.
                    userChoice = int(input(diner.getName() + ", please select a " + menuCategory + " menu item number"
                    + ".\n > "))  # Then ask the diner to order a menu item by selecting a number.
                    while userChoice >= self.menu.getNumMenuItems(menuCategory) or userChoice < 0:
                        # For error checking, make sure that the user enters a valid integer.
                        # Use the appropriate Menu method to get the number of menu items based on the category.
                        print("Invalid choice.")
                        userChoice = int(input(diner.getName() + ", please select a valid " + menuCategory + " menu item"
                         + " number.\n > "))
                    itemToAdd = self.menu.getMenuItem(menuCategory, userChoice)
                    diner.addToOrder(itemToAdd)
                diner.printOrder()

    def printMealCost(self):
        """
        :param: None
        :return: None
        """
        for diner in self.diners:  # Loop through the list of diners
            if diner.getStatus() == 3:  # check if the diner’s status is “paying”.
                payingSum = diner.getMealCost()  # For each diner that is paying and calculate the
                # diner’s meal cost by calling the appropriate Diner method.
                print(diner.getName() + ", your meal cost is $" + "{:.2f}".format(payingSum))  # Print out the total
                # cost in a message to the diner.
        print()

    def removeDiners(self):
        """
        :param: None
        :return:None
        """
        for diner in range(len(self.diners) - 1, -1, -1):  # loop through diners backwards using a range-based for loop:
            dinerObject = self.diners[diner]
            if dinerObject.getStatus() == 4:  # check if the diner’s status is “leaving”.
                print(dinerObject.getName() + ", thank you for dining with us! Come again soon!")  # For each diner
                # that is leaving, print a message thanking the diner.
                self.diners.remove(self.diners[diner])  # remove diner

    def advanceDiners(self):
        """
        :param: None
        :return: None
        """
        # This method allows the waiter to attend to the diners at their various stages as well as move each diner
        # on to the next stage.
        self.printDinerStatuses()  # First, call the printDinerStatuses() method.
        self.takeOrders()  # Then, in order, call takeOrders()
        self.printMealCost()  # printMealCost()
        self.removeDiners()  # and removeDiners()
        for diner in self.diners:  # Finally, update each diner’s status by looping through the list of diners
            diner.updateStatus()  # and calling the appropriate Diner method.


# def main():
#     menu = Menu("menu.csv")
#     waiter = Waiter(menu)
#     diner = Diner("Mehr")
#     waiter.addDiner(diner)
#     while waiter.getNumDiners() > 0:
#         waiter.advanceDiners()
#
#
# main()
