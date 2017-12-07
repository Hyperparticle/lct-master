all: wellformed valid

wellformed:
	xmllint --noout --valid person.xml

valid:
	xmllint --noout --dtdvalid person.dtd person.xml
