class MenuItem:  # Start by defining the MenuItem class in a file titled MenuItem.py.
# This class will represent a single item that a diner can order from the restaurant’s menu.

    def __init__(self, nameParam, categoryParam, priceParam, descParam):  # Class includes these instance attributes:
        self.name = nameParam  # a string representing the name of the MenuItem
        self.category = categoryParam  # a string representing the category of the item
        self.price = float(priceParam)  # a float representing the price of the item
        self.desc = descParam  # a string containing a description of the item
        # Return value: none

# Define get methods for all the instance attributes.
    def getName(self):
        return self.name

    def getCategory(self):
        return self.category

    def getPrice(self):
        return self.price

    def getDesc(self):
        return self.desc

    def __str__(self):  # __str__
        # Construct a message containing all 4 attributes, formatted in a readable manner such as:
        # Name (Type): $Price<new line> <tab>Description
        return self.name + " (" + self.category + "): $" + "{:.2f}".format(self.price) + "\n\t" + self.desc

# Unit test
# def main():
#     menuItem = MenuItem("Apple Pie", "Dessert", 7.0, "Cinnamon apples on our flaky crust.")
#     name = menuItem.getName()
#     print("name =", name)
#     price = menuItem.getPrice()
#     print("price =", price)
#     print(menuItem)
#
# main()
