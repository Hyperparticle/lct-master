all: wellformed valid

wellformed:
	xmllint --noout person.xml

valid:
	xmllint --noout --dtdvalid person.dtd person.xml
