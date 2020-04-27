//
//  ContentView.swift
//  URLCaller
//
//  Created by Junho HA on 2019/12/10.
//  Copyright © 2019 TAKIT. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    
    var body: some View {
        Button(action: {
            btnClick()
        }) {
             Text("Open anotherApp")
             }
    }
}

func btnClick() {
    print("onClickBTN")
    guard let s = URL(string : "url://asdfa") else {
        return }
    
    if UIApplication.shared.canOpenURL(s) {
        UIApplication.shared.open(s, options: [:]) { (b ) in
            print(b)
        }
    }
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
