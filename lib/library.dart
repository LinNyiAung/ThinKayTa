import 'package:flutter/material.dart';

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: LibraryPage(),
  ));
}

class LibraryPage extends StatelessWidget {
  // Define your items with their respective onPressed functions
  @override
  Widget build(BuildContext context) {
    final List<Map<String, dynamic>> items = [
      {
        "title": "Alphabets",
        "subtitle": "Videos and Images",
        "onPressed": () {
          Navigator.pushNamed(context, '/alphabet');
        },
      },
      {
        "title": "Words",
        "subtitle": "Videos and Images",
        "onPressed": () {
          Navigator.pushNamed(context, '/word');
        },
      },
      {
        "title": "Courses",
        "subtitle": "Videos and Images",
        "onPressed": () {
          Navigator.pushNamed(context, '/course');
        },
      },
    ];

    return Scaffold(
      backgroundColor:
          Color(0xffE8083E), // Background color to match your design
      body: Column(
        children: [
          SizedBox(height: 20), // Top padding
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Image.asset(
              'assets/logonobg.png', // Replace with your logo asset path
              height: 140,
            ),
          ),
          SizedBox(height: 20),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
              ),
            ),
          ),
          SizedBox(height: 20),
          Expanded(
            child: ListView.builder(
              padding: EdgeInsets.symmetric(horizontal: 20),
              itemCount: items.length,
              itemBuilder: (context, index) {
                return Container(
                  margin: EdgeInsets.only(bottom: 35),
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.all(15),
                      backgroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                      ),
                    ),
                    onPressed: items[index]
                        ['onPressed'], // Use the onPressed here
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          items[index]['title'],
                          style: TextStyle(
                            color: Colors.black,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 5),
                        Text(
                          items[index]['subtitle'],
                          style: TextStyle(
                            color: Colors.black54,
                            fontSize: 14,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          Container(
            color: Color(0xffBB0032),
            height: 100,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                IconButton(
                  icon: Icon(Icons.home, color: Colors.white, size: 30),
                  onPressed: () {
                    Navigator.pushNamed(context, "/home");
                  },
                ),
                IconButton(
                  icon: Icon(Icons.camera_alt, color: Colors.white, size: 30),
                  onPressed: () {
                    Navigator.pushNamed(context, "/asl");
                  },
                ),
                Container(
                  color: Colors.grey.withOpacity(0.7),
                  child: IconButton(
                    icon: Icon(Icons.book, color: Colors.white, size: 30),
                    onPressed: () {
                      Navigator.pushNamed(context, "/library");
                    },
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.settings, color: Colors.white, size: 30),
                  onPressed: () {
                    Navigator.pushNamed(context, "/setting");
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
