//
//  jhFile.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit


protocol Valuable {
    func value() -> String
}

class Value: Valuable {
    private let v: String
    
    init(value: String) {
        v = value
    }
    
    func value() -> String {
        return v
    }
}

class jhFile {
    
    init() {}
    
    static func converterToArray<T>(_ res: String, _ ext: String, _ arr : T.Type) -> Array<T> {
        var dataSource : Array<T> = Array()
        
        let lvPath = Bundle.main.url(forResource: res, withExtension: ext)!
        let lvArray = try! Data(contentsOf: lvPath)
        let decoder = PropertyListDecoder()
        if T.self == Int.self {
            dataSource = try! decoder.decode(Array<Int>.self, from: lvArray) as! Array<T>
            if GS.s.logLevel.contains(.graph) { print("jhFile_Array<Int>dataSource => ", dataSource) }
        } else if T.self == plistV1.self {
            dataSource = try! decoder.decode(Array<plistV1>.self, from: lvArray) as! Array<T>
            if GS.s.logLevel.contains(.graph) { print("jhFile_Array<plistV1>dataSource count => ", (dataSource as! [plistV1]).count) }
        } else if T.self == String.self {
            dataSource = try! decoder.decode(Array<String>.self, from: lvArray) as! Array<T>
            if GS.s.logLevel.contains(.graph) { print("jhFile_Array<String>dataSource => ", dataSource) }
        } else {
            dataSource = Array()
        }
        
        return dataSource
    }
    
    static func legacyConverterToArray(_ res: String, _ ext: String) -> NSArray? {
        let vPath = Bundle.main.path(forResource: res, ofType: ext)
        let retArray = NSArray(contentsOfFile: vPath!)
        return retArray
    }
}
