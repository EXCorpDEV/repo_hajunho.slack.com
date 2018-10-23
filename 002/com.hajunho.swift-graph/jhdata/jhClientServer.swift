//
//  jhClientServer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 18..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhClientServer {
    
    public static var mValuesOfDatas : Array<CGFloat> = Array() {
        didSet {
            if GS.shared.logLevel.contains(.graph) {
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas.count) in jhPanel")
            }
        }
    }
    
    public static var mCountOfaxes_view : Int = 1
    
    private static var listeners = [observer_p]()
    
    static func attachObserver(observer : observer_p) {
        listeners.append(observer)
    }

    /// not yet
    public static func notiDataDowloadFinish() {
//        for x in listeners {
//            x.jhRedraw()
//        }
    }
    
    //    public func getData() -> Array<CGFloat> {
    //        return self.mValuesOfDatas
    //    }
    //
    //    public func setData(x: Array<CGFloat>) {
    //        self.mValuesOfDatas = x
    //    }
}
