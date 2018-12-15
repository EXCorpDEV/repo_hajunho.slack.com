//
//  ViewController.swift
//
//  Created by Junho HA on 05/12/2018.
//  Copyright © 2018 M. All rights reserved.
//

import UIKit
import SwiftSocket

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
//        private String mIP = "192.168.0.4";
//        private int mPort = 8888;
        
        let client = TCPClient(address: "192.168.49.1", port: 38388)
     
        switch client.connect(timeout: 10) {
        case .success:
        // Connection successful 🎉
            print("success")
        case .failure(let error):
            print("fail")
            // 💩
        }
//
//        var (success, errmsg) = client.connect(timeout: 10)
//        if success {
//
//            var (success, errmsg) = client.send(str:"GET / HTTP/1.0\n\n" )
//            if success {
//
//                var data = client.read(1024*10)
//                if let d = data {
//                    if let str = String.stringWithBytes(d, length: d.count, encoding: NSUTF8StringEncoding){
//                        print(str)
//                    }
//                }
//            }else {
//                print(errmsg)
//            }
//        } else {
//            print(errmsg)
//        }
//
    }


}

